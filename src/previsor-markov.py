import pandas as pd
import statsmodels.api as sm
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Ignorar avisos
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', UserWarning)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

print("--- 1. Carregando e Preparando os Dados ---")

# Carregar o dataset
try:
    df = pd.read_csv("dataset-final-final.csv")
except FileNotFoundError:
    print("Erro: Arquivo 'dataset-final-final.csv' não encontrado.")
    exit()

# Converter a coluna 'date' para datetime e definir como índice
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
df = df.sort_index()

# --- 2. Definindo Variáveis Alvo e Features Originais ---

target_col = 'Y_Target_Spread_Ponderado'
endog = df[target_col]

exog_cols = [
    'X_Aave_Utilization', 'X_USDC_Depeg_Risk', 'X_VIX', 'X_DGS2',
    'X_Gas_Price_Gwei', 'X_ETH_Returns', 'X_ETH_Vol_7D', 'X_BTC_Returns',
    'X_BTC_Vol_7D', 'X_Gas_MA_7D', 'X_VIX_Pct_Change',
    'compound_utilization_rate', 'Spread_Aave_Compound', 'delta_total_defi_tvl_7d'
]

# Criar o dataframe de features com constante (AINDA NÃO ATRASADO)
exog_with_const_original = sm.add_constant(df[exog_cols], prepend=True)

# --- 3. A MUDANÇA CRÍTICA: Atrasando as Features (Lagging) ---

print("Atrasando as features (X) em 1 período (dia)...")
# Atrasamos todas as features em 1 dia.
# O modelo aprenderá a prever Y(T) usando X(T-1)
exog_lagged = exog_with_const_original.shift(1)

# --- 4. Alinhando os Dados (Removendo NaNs) ---
# Juntamos o alvo (Y) com as features atrasadas (X_lagged)
# e removemos a primeira linha, que agora contém NaNs
df_model = endog.to_frame(name=target_col).join(exog_lagged)
df_model = df_model.dropna()

# Separar os dados finais para o modelo
endog_final = df_model[target_col]
exog_final = df_model.drop(columns=[target_col])

print(f"Dados prontos. {len(endog_final)} observações alinhadas.")

# --- 5. Construindo e Treinando o Modelo PREVISOR ---

print("\n--- 5. Treinando Modelo Markov-Switching PREVISOR (k=3) ---")

# Definir quais coeficientes mudam (apenas o intercepto, 'const')
switching_exog_list = [True] + [False] * len(exog_cols)

model_predictor = sm.tsa.MarkovRegression(
    endog_final, 
    k_regimes=3,
    exog=exog_final,
    switching_variance=True,
    switching_exog=switching_exog_list
)

try:
    result_predictor = model_predictor.fit(maxiter=200, search_reps=10)
    
    print("\n--- 6. Modelo PREVISOR Treinado com Sucesso ---")
    print(result_predictor.summary())

    # --- 7. COMO FAZER A PREVISÃO (Exemplo) ---
    print("\n\n--- 8. Exemplo de Previsão para T+1 (Amanhã) ---")
    
    # 1. Pegar os dados de "hoje" (o último dia no dataset original)
    # Estes são os X(T) que preverão Y(T+1)
    today_features = exog_with_const_original.iloc[[-1]]
    
    # 2. Chamar o método .forecast()
    # O modelo usará as features de "hoje" (today_features)
    # e as probabilidades do último dia para prever o próximo.
    forecast = result_predictor.forecast(steps=1, exog=today_features)
    
    # O 'forecast' contém a média prevista e as probabilidades
    predicted_mean = forecast['mean'].iloc[0]
    predicted_probs = forecast['marginal_probabilities'].iloc[0]
    
    print(f"Features de 'hoje' ({today_features.index[0].date()}) usadas para a previsão.")
    print("\nPrevisão para o próximo dia (T+1):")
    print("------------------------------------------")
    print(f"**Spread Ponderado Previsto:** {predicted_mean:.4f}")
    print("\nProbabilidades do Regime Previsto:")
    
    # Vamos nomear os regimes com base no que aprendemos
    # (Isso assume que o modelo os ordenou da mesma forma,
    # precisamos checar o 'const' no summary() para confirmar)
    # Assumindo a mesma ordem do modelo anterior (Regime 2 = Fuga, 0 = Calmo, 1 = Euforia)
    print(f"  - P(Regime Fuga/Medo):   {predicted_probs[2]*100:.2f}%")
    print(f"  - P(Regime Calmo):        {predicted_probs[0]*100:.2f}%")
    print(f"  - P(Regime Euforia/Pico): {predicted_probs[1]*100:.2f}%")


except Exception as e:
    print(f"\nErro ao treinar o Modelo: {e}")