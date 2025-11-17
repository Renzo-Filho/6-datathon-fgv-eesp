import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- 1. Carregar os Dados ---
df = pd.read_csv("../data/dataset_final.csv")
df = df.rename(columns={'Unnamed: 0' : 'date'})
df.head()

# --- 2. Preparar os Dados ---
# Converter a coluna 'date' de string para datetime e defini-la como índice
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# --- 3. Gerar "O Gráfico da Tese" (Plot de Série Temporal) ---
plt.figure(figsize=(14, 7))
ax = sns.lineplot(x=df.index, y=df['Y_Target_Spread'], label='Spread (Aave - Ondo)')

# Adicionar uma linha de base em 0
ax.axhline(0, ls='--', color='red', lw=1.5, label='Spread = 0 (Sem Prêmio de Risco)')

# Melhorar a formatação
plt.title('O Gráfico da Tese: Spread de Risco DeFi (Aave - Ondo) ao Longo do Tempo', fontsize=16)
plt.ylabel('Spread de Risco (% APY)', fontsize=12)
plt.xlabel('Data', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Plotar o gráfico
plt.show() # <-- CORREÇÃO: Usar plt.show() para exibir o gráfico

# --- 4. Engenharia de Features ---
df['Y_Target_Diff'] = df['Y_Target_Spread'].diff()
df['Lag_1'] = df['Y_Target_Spread'].shift(1)
df['Lag_2'] = df['Y_Target_Spread'].shift(2)
df['Diff_Lag_1'] = df['Y_Target_Diff'].shift(1)

# --- APLICANDO SUGESTÃO 2: Transformações ---
df['X_BTC_Ret'] = df['X_BTC_Price'].pct_change()
df['X_Gas_Change'] = df['X_Gas_Price_Gwei'].diff()

df = df.dropna()

# Seleção de Features (Novas features engenheiradas)
features = [
    'X_Aave_Utilization', 'totalValueLockedUSD', 'X_USDC_Depeg_Risk', 'X_VIX',
    'X_BTC_Vol_7D', 'X_ETH_Vol_7D', 'X_ETH_Returns', 'X_DGS2',
    'Lag_1', 'Lag_2', 'Diff_Lag_1', 'X_BTC_Ret', 'X_Gas_Change'
]

# A lista 'features_sem' estava causando o KeyError
# features_sem = [
#     'totalValueLockedUSD', 'X_USDC_Depeg_Risk', 'X_VIX',
#     'X_BTC_Vol_7D', 'X_ETH_Vol_7D', 'X_ETH_Returns', 'X_DGS2',
#     'Lag_1', 'Lag_2', 'Diff_Lag_1', 'X_BTC_Ret', 'X_Gas_Change'
# ]

# <-- CORREÇÃO: Usar 'features' para que X_Aave_Utilization esteja incluída
X = df[features] 
Y = df['Y_Target_Diff']
#Y = df['Y_Target_Spread']

# --- 5. Divisão Treino/Teste (Time Series) ---
# NUNCA usar shuffle=True em séries temporais
split_point = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
Y_train, Y_test = Y.iloc[:split_point], Y.iloc[split_point:]

print(f"--- Divisão dos Dados (Time Series Split) ---")
print(f"Total de amostras: {len(df)}")
print(f"Amostras de Treino: {len(Y_train)} (80%)")
print(f"Amostras de Teste:  {len(Y_test)} (20%)")

# <-- CORREÇÃO: Removida a função 'get_time_period_data' e suas chamadas,
# pois não estavam sendo usadas e continham erros.

# --- 6. Clusterização por Regimes (Método Corrigido) ---

def regime_based_clustering(X, volatility_threshold=0.7, utilization_threshold=0.7, 
                            vol_thresh=None, util_thresh=None): # <-- CORREÇÃO: Adicionados thresholds opcionais
    """
    Clusterização baseada em regimes de mercado.
    Se vol_thresh e util_thresh não forem fornecidos, eles são "aprendidos" dos dados X.
    Se forem fornecidos, eles são "aplicados" aos dados X.
    """
    regimes = np.zeros(len(X))
    
    # --- "Fit" (Aprender) thresholds se não forem fornecidos ---
    if vol_thresh is None:
        vol_thresh = X['X_VIX'].quantile(volatility_threshold)
        print(f"Threshold VIX (Aprendido): {vol_thresh:.2f}")
    else:
        print(f"Threshold VIX (Aplicado): {vol_thresh:.2f}")

    if util_thresh is None:
        util_thresh = X['X_Aave_Utilization'].quantile(utilization_threshold)
        print(f"Threshold Aave Util (Aprendido): {util_thresh:.2f}")
    else:
        print(f"Threshold Aave Util (Aplicado): {util_thresh:.2f}")

    # Definir regimes
    normal_mask = (X['X_VIX'] <= vol_thresh) & (X['X_Aave_Utilization'] <= util_thresh)
    stress_mask = ((X['X_VIX'] > vol_thresh) | (X['X_Aave_Utilization'] > util_thresh)) & \
                  ~((X['X_VIX'] > vol_thresh) & (X['X_Aave_Utilization'] > util_thresh))
    crisis_mask = (X['X_VIX'] > vol_thresh) & (X['X_Aave_Utilization'] > util_thresh)
    
    regimes[normal_mask] = 0
    regimes[stress_mask] = 1
    regimes[crisis_mask] = 2
    
    # Estatísticas por regime
    regime_stats = pd.DataFrame({
        'Regime': ['Normal', 'Estresse', 'Crisis'],
        'Samples': [np.sum(normal_mask), np.sum(stress_mask), np.sum(crisis_mask)],
        'Avg_VIX': [X[normal_mask]['X_VIX'].mean(), X[stress_mask]['X_VIX'].mean(), X[crisis_mask]['X_VIX'].mean()],
        'Avg_Utilization': [X[normal_mask]['X_Aave_Utilization'].mean(), X[stress_mask]['X_Aave_Utilization'].mean(), X[crisis_mask]['X_Aave_Utilization'].mean()]
    })
    
    print("\nDistribuição dos Regimes:")
    print(regime_stats)
    
    # <-- CORREÇÃO: Retornar os regimes e os thresholds aprendidos
    return regimes, vol_thresh, util_thresh

# --- 7. Treino e Avaliação dos Modelos por Regime ---

print("=== MÉTODO 2: Clusterização por Regimes ===\n")

# --- CORREÇÃO: Aplicar lógica de Fit/Transform ---
# 1. "Fit": Aprender os regimes e thresholds *APENAS* no treino
print("--- Definindo Regimes (TREINO) ---")
train_regimes, v_thresh, u_thresh = regime_based_clustering(X_train, 
                                                            volatility_threshold=0.7, 
                                                            utilization_threshold=0.7)

# 2. "Transform": Aplicar os thresholds *aprendidos no treino* aos dados de teste
print("\n--- Aplicando Regimes (TESTE) ---")
test_regimes, _, _ = regime_based_clustering(X_test, 
                                             vol_thresh=v_thresh, 
                                             util_thresh=u_thresh)
# --- Fim da Correção ---


# Treinar modelos por regime
regime_models = {}
regime_predictions = np.zeros(len(X_test))
regime_metrics = {}

print("\n--- Treinando Modelos por Regime ---")
for regime_id in [0, 1, 2]:
    # Dados de treino para este regime
    train_mask = train_regimes == regime_id
    
    if np.sum(train_mask) > 10:  # Mínimo de amostras
        print(f"\nTreinando Modelo para Regime {regime_id} (Amostras: {np.sum(train_mask)})...")
        X_regime_train = X_train[train_mask]
        y_regime_train = Y_train[train_mask]
        
        # Treinar LASSO para o regime
        model = LassoCV(alphas=np.logspace(-4, 0, 50), cv=3, random_state=42)
        model.fit(X_regime_train, y_regime_train)
        regime_models[regime_id] = model
        
        # Prever para amostras de teste no mesmo regime
        test_mask = test_regimes == regime_id
        if np.sum(test_mask) > 0:
            X_regime_test = X_test[test_mask]
            y_regime_test = Y_test[test_mask]
            
            y_pred_regime = model.predict(X_regime_test)
            regime_predictions[test_mask] = y_pred_regime
            
            # Calcular métricas por regime
            mae_regime = mean_absolute_error(y_regime_test, y_pred_regime)
            rmse_regime = np.sqrt(mean_squared_error(y_regime_test, y_pred_regime))
            r2_regime = r2_score(y_regime_test, y_pred_regime)
            
            regime_metrics[regime_id] = {
                'samples': np.sum(test_mask),
                'mae': mae_regime,
                'rmse': rmse_regime,
                'r2': r2_regime
            }
            
            print(f"Regime {regime_id} - Amostras teste: {np.sum(test_mask)}")
            print(f"MAE: {mae_regime:.4f}, RMSE: {rmse_regime:.4f}, R²: {r2_regime:.4f}")
        else:
            print(f"Regime {regime_id} - Sem amostras de teste.")
    else:
        print(f"\nRegime {regime_id} - Pulado (amostras de treino insuficientes: {np.sum(train_mask)})")


# Métricas gerais
# Nota: A métrica geral pode ser enganosa se alguns regimes não tiverem previsões
valid_test_mask = (test_regimes == 0) | (test_regimes == 1) | (test_regimes == 2)
overall_mae = mean_absolute_error(Y_test[valid_test_mask], regime_predictions[valid_test_mask])
overall_rmse = np.sqrt(mean_squared_error(Y_test[valid_test_mask], regime_predictions[valid_test_mask]))
overall_r2 = r2_score(Y_test[valid_test_mask], regime_predictions[valid_test_mask])

print(f"\n=== PERFORMANCE GERAL (Regimes) === \n(Apenas sobre amostras de teste com regimes mapeados)")
print(f"MAE: {overall_mae:.4f}")
print(f"RMSE: {overall_rmse:.4f}")
print(f"R²: {overall_r2:.4f}")

# Visualizar regimes
plt.figure(figsize=(12, 6))
colors = ['green', 'orange', 'red']
regime_names = ['Normal (0)', 'Estresse (1)', 'Crisis (2)']

for regime_id in [0, 1, 2]:
    mask = test_regimes == regime_id
    if np.sum(mask) > 0:
        # Usar o índice de X_test para obter a posição correta
        test_indices = np.where(mask)[0]
        plt.scatter(test_indices, Y_test[mask], 
                   c=colors[regime_id], label=regime_names[regime_id], alpha=0.6, s=15)

plt.xlabel('Índice da Amostra de Teste')
plt.ylabel('Y_Target_Diff Real')
plt.title('Distribuição dos Regimes no Conjunto de Teste')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()