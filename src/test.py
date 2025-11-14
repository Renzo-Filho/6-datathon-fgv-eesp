from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# --- 1. Treinamento dos Modelos ---
predictions = {}

# Modelo 1: Baseline (Ingênuo)
# Previsão é o valor anterior (shift)
y_pred_baseline = Y_test.shift(1)
y_pred_baseline.iloc[0] = Y_train.iloc[-1]
predictions['Baseline'] = y_pred_baseline

# Modelo 2: Linear Regression
model_lr = LinearRegression()
model_lr.fit(X_train, Y_train)
predictions['Linear Regression'] = pd.Series(model_lr.predict(X_test), index=Y_test.index)

# Modelo 3: XGBoost
model_xgb = XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=100)
model_xgb.fit(X_train, Y_train)
predictions['XGBoost'] = pd.Series(model_xgb.predict(X_test), index=Y_test.index)

# Modelo 4: Random Forest
model_rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
model_rf.fit(X_train, Y_train)
predictions['Random Forest'] = pd.Series(model_rf.predict(X_test), index=Y_test.index)

# --- 2. Cálculo de Métricas e Seleção dos Top 3 ---
metrics_list = []
for name, y_pred in predictions.items():
    mae = mean_absolute_error(Y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(Y_test, y_pred))
    r2 = r2_score(Y_test, y_pred)
    metrics_list.append({'Model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2})

# Criar DataFrame e ordenar pelo menor RMSE
df_metrics = pd.DataFrame(metrics_list).sort_values(by='RMSE', ascending=True)
top_3_models = df_metrics.head(3)['Model'].tolist()

print("--- Ranking dos Modelos (por RMSE) ---")
print(df_metrics.to_markdown(index=False))
print(f"\nTop 3 Modelos selecionados para plotagem: {top_3_models}")

# --- 3. Plotagem dos Gráficos Comparativos (Top 3) ---
sns.set_style("whitegrid")
plt.figure(figsize=(20, 12))

# Gráfico 1: Série Temporal (Real vs Top 3 Previstos)
plt.subplot(2, 2, 1)
plt.plot(Y_test.index, Y_test, label='Real', color='black', linewidth=2, alpha=0.8)
colors = ['green', 'blue', 'red'] # Cores para o top 3
for i, model_name in enumerate(top_3_models):
    plt.plot(Y_test.index, predictions[model_name], label=model_name, 
             linestyle='--', alpha=0.7, color=colors[i])
plt.title(f'Comparação Temporal: Real vs Top 3 Modelos', fontsize=14)
plt.ylabel('Spread Diff')
plt.legend()

# Gráfico 2: Comparação de RMSE (Barra)
plt.subplot(2, 2, 2)
sns.barplot(x='RMSE', y='Model', data=df_metrics.head(3), palette='viridis')
plt.title('Comparação de RMSE (Menor é Melhor)', fontsize=14)
plt.xlabel('RMSE')

# Gráfico 3: Dispersão (Scatter Plot) Real vs Previsto
plt.subplot(2, 2, 3)
for i, model_name in enumerate(top_3_models):
    plt.scatter(Y_test, predictions[model_name], label=model_name, alpha=0.5, color=colors[i], s=15)
# Linha ideal
min_val = min(Y_test.min(), min([predictions[m].min() for m in top_3_models]))
max_val = max(Y_test.max(), max([predictions[m].max() for m in top_3_models]))
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Ideal')
plt.title('Dispersão: Real vs Previsto', fontsize=14)
plt.xlabel('Valor Real')
plt.ylabel('Valor Previsto')
plt.legend()

# Gráfico 4: Distribuição dos Resíduos (KDE)
plt.subplot(2, 2, 4)
for i, model_name in enumerate(top_3_models):
    residuals = Y_test - predictions[model_name]
    sns.kdeplot(residuals, label=f'{model_name} (Resíduos)', fill=True, alpha=0.2, color=colors[i])
plt.axvline(0, color='black', linestyle='--')
plt.title('Distribuição dos Erros (Resíduos)', fontsize=14)
plt.xlabel('Erro (Real - Previsto)')
plt.legend()

plt.tight_layout()
plt.show()