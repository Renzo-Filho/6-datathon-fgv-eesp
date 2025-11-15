import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class Model:
    
    def __init__(self, volatility_threshold=0.7, utilization_threshold=0.7):
        """
        Inicializa o modelo, carrega os dados, divide em treino/teste
        e treina os modelos por regime.
        """
        print("Iniciando o Modelo...")
        
        # --- 1. Armazenamento ---
        # Modelos e scalers serão armazenados por regime_id (0, 1, 2)
        self.models = {}
        self.v_thresh = None
        self.u_thresh = None
        self.volatility_threshold = volatility_threshold
        self.utilization_threshold = utilization_threshold
        
        # --- 2. Carregar Dados ---
        try:
            X = pd.read_csv("X.csv", index_col='date', parse_dates=True)
            Y = pd.read_csv("Y.csv", index_col='date', parse_dates=True)['Y_Target_Spread']
        except FileNotFoundError:
            print("Erro: X.csv ou Y.csv não encontrados. Execute o notebook datathon-cluster.ipynb primeiro.")
            return

        self.features = X.columns.tolist()

        # --- 3. Divisão Treino/Teste ---
        split_point = int(len(X) * 0.8)
        self.X_train, self.X_test = X.iloc[:split_point], X.iloc[split_point:]
        self.Y_train, self.Y_test = Y.iloc[:split_point], Y.iloc[split_point:]

        print(f"Dados carregados: {len(X)} amostras.")
        print(f"Treino: {len(self.Y_train)}, Teste: {len(self.Y_test)}")

        # --- 4. Treinar Modelos ---
        self._train()
        
        # --- 5. Preparar dados para plotagem ---
        self._prepare_plot_data()
        print("Modelos treinados e prontos.")


    def _regime_based_clustering(self, X, vol_thresh=None, util_thresh=None): 
        """
        Helper privado para clusterização baseada em regimes.
        Modo "Fit": Se vol_thresh/util_thresh são None, aprende novos thresholds.
        Modo "Transform": Se vol_thresh/util_thresh são fornecidos, aplica-os.
        """
        regimes = np.zeros(len(X), dtype=int)    
        
        # Modo "Fit" (Aprender)
        if vol_thresh is None:
            vol_thresh = X['X_VIX'].quantile(self.volatility_threshold)
            util_thresh = X['X_Aave_Utilization'].quantile(self.utilization_threshold)
            print(f"--- Modo Fit (Treino) ---")
            print(f"Threshold VIX (Aprendido): {vol_thresh:.2f}")
            print(f"Threshold Aave Util (Aprendido): {util_thresh:.2f}")
        # Modo "Transform" (Aplicar)
        else:
            print(f"--- Modo Transform (Teste/Previsão) ---")
            print(f"Threshold VIX (Aplicado): {vol_thresh:.2f}")
            print(f"Threshold Aave Util (Aplicado): {util_thresh:.2f}")

        # Definir 3 regimes
        normal_mask = (X['X_VIX'] <= vol_thresh) & (X['X_Aave_Utilization'] <= util_thresh)
        crisis_mask = (X['X_VIX'] > vol_thresh) & (X['X_Aave_Utilization'] > util_thresh)
        
        # Estresse é qualquer um acima, mas não ambos
        stress_mask = ((X['X_VIX'] > vol_thresh) | (X['X_Aave_Utilization'] > util_thresh)) & \
                      ~(crisis_mask)
        
        regimes[normal_mask] = 0  # Normal
        regimes[stress_mask] = 1  # Estresse
        regimes[crisis_mask] = 2  # Crise
        
        return regimes, vol_thresh, util_thresh

    def _train(self):
        """
        Treina os modelos para cada regime usando os dados de self.X_train.
        """
        print("\nTreinando modelos por regime...")
        
        # 1. "Fit" (Aprender) thresholds e regimes APENAS no treino
        (
            self.train_regimes, 
            self.v_thresh, 
            self.u_thresh
        ) = self._regime_based_clustering(self.X_train)

        # 2. "Transform" (Aplicar) thresholds nos dados de teste (para plotagem e avaliação)
        (
            self.test_regimes, 
            _, 
            _
        ) = self._regime_based_clustering(self.X_test, 
                                          vol_thresh=self.v_thresh, 
                                          util_thresh=self.u_thresh)

        # 3. Definir parâmetros dos modelos
        
        # Parâmetros otimizados para o Regime Normal (como você encontrou)
        rf_params_normal = {
            'max_depth': None, 
            'max_features': 0.8, 
            'min_samples_leaf': 1, 
            'min_samples_split': 10, 
            'n_estimators': 100,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Parâmetros padrão para os outros regimes (você pode otimizá-los depois)
        rf_params_default = {
            'random_state': 42,
            'n_jobs': -1
        }

        # 4. Treinar um modelo para cada regime
        for regime_id in [0, 1, 2]:
            train_mask = self.train_regimes == regime_id
            X_regime_train = self.X_train[train_mask]
            y_regime_train = self.Y_train[train_mask]

            if len(X_regime_train) < 10:
                print(f"Regime {regime_id}: Pulado (amostras de treino insuficientes: {len(X_regime_train)})")
                continue
            
            print(f"Treinando Modelo para Regime {regime_id} (Amostras: {len(X_regime_train)})...")
            
            if regime_id == 0:
                model = RandomForestRegressor(**rf_params_normal)
            else:
                # Usar padrão para regime 1 (Estresse) e 2 (Crise)
                model = RandomForestRegressor(**rf_params_default)
            
            model.fit(X_regime_train, y_regime_train)
            self.models[regime_id] = model

    def predict(self, X_new):
        """
        Prevê o spread para novos dados (X_new) usando os modelos de regime treinados.
        
        Args:
            X_new (pd.DataFrame): DataFrame com as features de entrada.
            
        Returns:
            pd.Series: Série com as previsões de spread.
        """
        if not self.models:
            raise ValueError("Modelos não foram treinados. Chame _train() primeiro (feito no __init__).")
            
        print(f"Prevendo para {len(X_new)} novas amostras...")
        
        # 1. Aplicar regimes aos novos dados usando os thresholds SALVOS
        regimes, _, _ = self._regime_based_clustering(
            X_new, 
            vol_thresh=self.v_thresh, 
            util_thresh=self.u_thresh
        )
        
        predictions = pd.Series(index=X_new.index, dtype=float)
        
        # 2. Iterar e aplicar o modelo correto
        for regime_id in [0, 1, 2]:
            mask = (regimes == regime_id)
            
            if mask.sum() > 0:
                X_subset = X_new[mask]
                
                # 3. Usar o modelo treinado. Se o modelo não existir (ex: 'Crise' sem treino),
                #    usamos o modelo 'Normal' (regime 0) como fallback.
                model = self.models.get(regime_id, self.models.get(0))
                
                if model is None:
                    # Isso só aconteceria se nem o regime 0 fosse treinado
                    raise ValueError("Nenhum modelo válido encontrado para predição.")

                print(f"Aplicando modelo do Regime {model.regime_id if hasattr(model, 'regime_id') else (0 if model == self.models.get(0) else regime_id)} para {mask.sum()} amostras.")
                
                pred_subset = model.predict(X_subset[self.features]) # Garante a ordem das colunas
                predictions[mask] = pred_subset
                
        return predictions
    
    def evaluate(self):
        """
        Avalia os modelos treinados contra os dados de teste (self.X_test).
        """
        preds = self.predict(self.X_test)
        
        # Calcular métricas
        mae = mean_absolute_error(self.Y_test, preds)
        rmse = np.sqrt(mean_squared_error(self.Y_test, preds))
        r2 = r2_score(self.Y_test, preds)
        
        print("\n--- Performance Geral (Teste) ---")
        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²:   {r2:.4f}")
        
        return {'MAE': mae, 'RMSE': rmse, 'R2': r2}, preds

    def _prepare_plot_data(self):
        """
        Cria o df_plot para visualização após o treino.
        """
        df_plot = pd.concat([self.X_train, self.X_test])
        df_plot['Y_Target_Spread'] = pd.concat([self.Y_train, self.Y_test])
        
        df_plot['Regime'] = 0 # Default
        df_plot.loc[self.X_train.index, 'Regime'] = self.train_regimes
        df_plot.loc[self.X_test.index, 'Regime'] = self.test_regimes
        
        df_plot['Regime'] = df_plot['Regime'].astype(int)
        self.df_plot = df_plot

    def view_clusters_plot(self):
        """
        Plota o gráfico de 'A Tese' com os regimes de cluster.
        (Copiado da Célula 312 do notebook e corrigido)
        """
        if self.df_plot is None:
            print("Dados de plotagem não preparados. Treine o modelo primeiro.")
            return

        plt.figure(figsize=(16, 8))
        ax = sns.lineplot(
            data=self.df_plot,
            x=self.df_plot.index,
            y='Y_Target_Spread',
            label='Spread (Aave - Ondo)',
            color='black',
            linewidth=1.0
        )
        ax.axhline(0, ls='--', color='red', lw=1.5, label='Spread = 0 (Sem Prêmio)')

        colors = ['#2ca02c', '#ff7f0e', '#d62728'] 
        regime_names = ['Regime 0: Normal', 'Regime 1: Estresse', 'Regime 2: Crise']
        plotted_labels = set() 

        start_date = self.df_plot.index[0]
        current_regime = self.df_plot.iloc[0]['Regime']

        for i in range(1, len(self.df_plot)):
            row = self.df_plot.iloc[i]
            
            if row['Regime'] != current_regime:
                end_date = self.df_plot.index[i]
                
                label = regime_names[int(current_regime)] if regime_names[int(current_regime)] not in plotted_labels else None        
                ax.axvspan(
                    start_date,
                    end_date,
                    color=colors[int(current_regime)],
                    alpha=0.2,
                    label=label,
                    zorder=0 
                )
                
                plotted_labels.add(regime_names[int(current_regime)])        
                start_date = end_date
                current_regime = row['Regime']

        # Desenhar o último bloco
        label = regime_names[int(current_regime)] if regime_names[int(current_regime)] not in plotted_labels else None
        ax.axvspan(
            start_date,
            self.df_plot.index[-1],
            color=colors[int(current_regime)],
            alpha=0.2,
            label=label,
            zorder=0
        )

        # Linha de Divisão Treino/Teste
        split_date = self.X_test.index[0]
        ax.axvline(
            split_date,
            color='blue',
            linestyle=':',
            linewidth=2,
            label='Início do Conjunto de Teste'
        )

        # Formatação
        plt.title('Spread de Risco com Regimes de Mercado (Clusters)', fontsize=18)
        plt.ylabel('Spread de Risco (% APY)', fontsize=12)
        plt.xlabel('Data', fontsize=12)
        
        try:
            handles, labels = ax.get_legend_handles_labels()
            # Tenta reordenar
            order = [i for i, lbl in enumerate(labels) if lbl.startswith('Regime')]
            order += [i for i, lbl in enumerate(labels) if not lbl.startswith('Regime')]
            ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper left', fontsize=11)
        except Exception:
            ax.legend(loc='upper left', fontsize=11)

        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()