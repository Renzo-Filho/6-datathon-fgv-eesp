# (Pode ser necessário: !pip install pytorch-lightning pytorch-forecasting)
import pandas as pd
import numpy as np
import torch
import lightning.pytorch as pl # Usar lightning.pytorch para garantir a versão 2.x
import matplotlib.pyplot as plt
torch.set_float32_matmul_precision('medium')
import os
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import RMSE
from sklearn.metrics import r2_score, mean_squared_error

if __name__ == "__main__":

    try:
        print("\nIniciando Treinamento (Teste 14 - TFT Regressão de Nível)...")

        # --- 1. Carregar Dados (Movido para dentro desta célula) ---
        file_path = 'dataset_final_completo_com_mkr.csv'
        print("Carregando e preparando dados...")
        df = pd.read_csv(file_path)
        target_col = 'Y_Target_Spread_Ponderado'
        df = df.drop(columns=['mkr_apyBase', 'mkr_apyReward'])
        df['date'] = pd.to_datetime(df['date'])
        df_clean = df.dropna(subset=[target_col]).copy()
        df_clean = df_clean.sort_values(by='date').reset_index(drop=True)

        # --- 2. Definir TODAS as Features (34) (Movido para dentro desta célula) ---
        market_risk_features = [
            'totalValueLockedUSD', 'X_USDC_Depeg_Risk', 'X_BTC_Price', 'X_ETH_Price',
            'X_VIX', 'X_DGS2', 'X_Gas_Price_Gwei', 'X_ETH_Returns', 'X_ETH_Vol_7D',
            'X_BTC_Returns', 'X_BTC_Vol_7D', 'X_Gas_MA_7D', 'X_VIX_Pct_Change',
            'total_tvl_aave', 'total_tvl_compound', 'total_tvl_defi',
            'delta_total_defi_tvl_7d', 'mkr_tvlUsd'
        ]

        autoregressive_features = [
            'Y_Aave_APY', 'Y_SOFR', 'Y_Target_Spread', 'DeFi_APY_Ponderado',
            'apy_aave', 'apy_compound', 'deposit_apy', 'borrow_apy', 'mkr_apy',
            'X_Aave_Utilization', 'utilization_rate', 'compound_utilization_rate',
            'total_borrow_usd', 'total_deposit_usd', 'Spread_Aave_Compound'
        ]

        all_features_to_model = market_risk_features + autoregressive_features
        feature_cols = [col for col in all_features_to_model if col in df_clean.columns]

        # --- 3. Criar DataFrame Final para os Modelos (Movido para dentro desta célula) ---
        X_y_df = df_clean[feature_cols + [target_col]].copy()
        for col in X_y_df.columns:
            X_y_df[col] = pd.to_numeric(X_y_df[col], errors='coerce')

        X_y_df = X_y_df.ffill().bfill()

        if X_y_df.isnull().sum().sum() > 0:
            print("Ainda há NaNs! Preenchendo com 0.")
            X_y_df = X_y_df.fillna(0)

        print(f"Dados prontos com {len(feature_cols)} features e 1 alvo.")

        # --- 1. Preparação de Dados para TFT ---
        tft_data = X_y_df.copy()

        # TFT requer uma coluna 'time_idx' (inteiro sequencial)
        tft_data['time_idx'] = np.arange(len(tft_data))

        # TFT requer uma coluna de 'group' (mesmo que só tenhamos um)
        tft_data['group'] = 0

        # --- 2. Divisão Cronológica (Treino/Teste) ---
        split_ratio = 0.8
        split_index = int(len(tft_data) * split_ratio)

        train_tft_df = tft_data.iloc[:split_index]
        test_tft_df = tft_data.iloc[split_index:]

        # --- 3. Criar o TimeSeriesDataSet ---
        n_lookback = 14 # Dias de histórico
        n_horizon = 1  # Prever 1 dia à frente

        n_workers = 8

        training_data = TimeSeriesDataSet(
            train_tft_df,
            time_idx="time_idx",
            target=target_col,
            group_ids=["group"],
            time_varying_unknown_reals=feature_cols,
            max_encoder_length=n_lookback,
            max_prediction_length=n_horizon,
            target_normalizer=GroupNormalizer(groups=["group"], method="robust"),
            add_relative_time_idx=True,
        )

        validation_data = TimeSeriesDataSet.from_dataset(
            training_data, tft_data, predict=False, stop_randomization=True
        )

        batch_size = 32
        
        train_dataloader = training_data.to_dataloader(
            train=True, batch_size=batch_size, num_workers=n_workers, persistent_workers=True
        )
        val_dataloader = validation_data.to_dataloader(
            train=False, batch_size=batch_size, num_workers=n_workers, persistent_workers=True
        )

        print("Dataloaders do TFT prontos.")

        # --- 4. Definir e Treinar o Modelo TFT ---
        trainer = pl.Trainer(
            max_epochs=300,
            accelerator="auto",
            gradient_clip_val=0.1,
            callbacks=[
                pl.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10) # Aumentei a paciência
            ],
        )

        tft = TemporalFusionTransformer.from_dataset(
            training_data,
            # NÃO DEFINA O LR AQUI AINDA
            hidden_size=128,
            attention_head_size=4,
            dropout=0.1,
            hidden_continuous_size=64,
            output_size=1,
            loss=RMSE(),
            reduce_on_plateau_patience=4,
        )
        
        # --- NOVO: Bloco do Tuner ---
        print("Iniciando o Tuner para encontrar o Learning Rate ideal...")
        tuner = pl.tuner.Tuner(trainer)
        
        # Encontra o LR
        lr_finder = tuner.lr_find(
            tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            max_lr=1.0,
            min_lr=1e-6,

        )

        # Pega a sugestão e plota (o plot é útil no notebook)
        suggested_lr = lr_finder.suggestion()
        print(f"Taxa de Aprendizado (LR) sugerida: {suggested_lr}")
        
        # Atualiza o modelo com o LR ideal
        tft.hparams.learning_rate = suggested_lr

        # --- Fim do Bloco do Tuner ---

        print(f"\nTreinando o modelo TFT com LR={suggested_lr} (parâmetros: {tft.size()})...")
        trainer.fit(
            tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )
        print("Treinamento TFT concluído.")

        # --- 5. Avaliação do TFT ---
        best_model_path = trainer.checkpoint_callback.best_model_path
        best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

        # Coletar as previsões (assume que .predict() retorna apenas o tensor de previsões)
        y_pred_tensor = best_tft.predict(val_dataloader)
        y_pred_tft = y_pred_tensor.flatten().numpy()

        # Coletar os valores reais do val_dataloader
        y_test_values_from_dataloader = []
        for x, y in val_dataloader:
            # y é uma tupla (target, encoder_target). Pegamos o target (primeiro elemento).
            y_test_values_from_dataloader.append(y[0].flatten())
        
        y_test_tft = torch.cat(y_test_values_from_dataloader).numpy()

        # Verifica se as dimensões correspondem antes de calcular as métricas
        if len(y_pred_tft) != len(y_test_tft):
            print(f"Aviso: Comprimento das previsões ({len(y_pred_tft)}) e dos valores reais ({len(y_test_tft)}) não correspondem. Alinhando.")
            min_len = min(len(y_pred_tft), len(y_test_tft))
            y_pred_tft = y_pred_tft[:min_len]
            y_test_tft = y_test_tft[:min_len]

        r2_tft = r2_score(y_test_tft, y_pred_tft)
        rmse_tft = np.sqrt(mean_squared_error(y_test_tft, y_pred_tft))

        print("\n--- Métricas (TFT) ---")
        print(f"R-squared (R²): {r2_tft:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse_tft:.6f}")


        # --- GRÁFICO 1: Previsão vs. Real (Série Temporal) ---
        # Este é o gráfico mais importante. Mostra como o modelo seguiu a realidade ao longo do tempo.

        plt.figure(figsize=(15, 6))
        plt.plot(y_test_tft, label='Valor Real', color='blue', linewidth=1)
        plt.plot(y_pred_tft, label='Previsão do Modelo', color='red', linestyle='--', linewidth=2)
        plt.title(f'TFT: Previsão vs. Real (R²: {r2_tft:.4f})')
        plt.xlabel('Amostras no Tempo (Conjunto de Teste)')
        plt.ylabel('Valor do Target (Spread Ponderado)')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()

        # Salva a imagem em um arquivo
        plt.savefig('tft_time_series_plot.png')
        print("Gráfico de Série Temporal salvo em 'tft_time_series_plot.png'")


        # --- GRÁFICO 2: Gráfico de Dispersão (Quão perfeito é o modelo?) ---
        # Se o modelo fosse perfeito, todos os pontos estariam na linha vermelha (y=x).

        plt.figure(figsize=(8, 8))
        # Limites para a linha y=x
        min_val = min(np.min(y_test_tft), np.min(y_pred_tft))
        max_val = max(np.max(y_test_tft), np.max(y_pred_tft))

        plt.scatter(y_test_tft, y_pred_tft, alpha=0.3, label='Previsões')
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Linha Perfeita (y=x)')
        plt.xlabel('Valores Reais')
        plt.ylabel('Valores Previstos')
        plt.title('Gráfico de Dispersão: Real vs. Previsto')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.axis('equal') # Garante que a escala de X e Y seja a mesma
        plt.tight_layout()

        # Salva a imagem em um arquivo
        plt.savefig('tft_scatter_plot.png')
        print("Gráfico de Dispersão salvo em 'tft_scatter_plot.png'")


    except Exception as e:
        print(f"Um erro ocorreu durante o treinamento do TFT: {e}")