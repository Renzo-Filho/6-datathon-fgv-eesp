import pandas as pd
import statsmodels.api as sm
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Ignorar avisos de convergência e usuário, que são comuns nestes modelos
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', UserWarning)

# Configurar pandas para ver todas as colunas
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

# --- 2. Definindo Variáveis Endógenas (Alvo) e Exógenas (Features) ---

# Variável Alvo (Y)
target_col = 'Y_Target_Spread_Ponderado'
endog = df[target_col]

# Features Explicativas (X)
# Mapeando os nomes do heatmap para os nomes de coluna reais no CSV
exog_cols = [
    'X_Aave_Utilization',
    'X_USDC_Depeg_Risk',
    'X_VIX',
    'X_DGS2',
    'X_Gas_Price_Gwei',
    'X_ETH_Returns',
    'X_ETH_Vol_7D',
    'X_BTC_Returns',
    'X_BTC_Vol_7D',
    'X_Gas_MA_7D',
    'X_VIX_Pct_Change',
    'compound_utilization_rate',  # Mapeado de X_Compound_Utilization
    'Spread_Aave_Compound',       # Mapeado de X_Spread_Aave_Compound
    'delta_total_defi_tvl_7d'     # Mapeado de X_Delta_TVL_DeFi_7D_pct
]

exog = df[exog_cols]

# Adicionar uma constante (intercepto) às variáveis exógenas
# Isso é crucial para o modelo
exog_with_const = sm.add_constant(exog, prepend=True)

print(f"Dados preparados. Alvo (Endog): {target_col}")
print(f"Features (Exog): {len(exog_cols)} colunas.")

# --- 3. Construindo e Treinando o Modelo Markov-Switching ---

print("\n--- 3. Treinando Modelo Markov-Switching (k=3) ---")

# Criar a lista que define o que muda
# A primeira feature (índice 0) é a 'const' (intercepto) que adicionamos
# Queremos que ela mude. Não queremos que as outras 14 mudem.
# [True, False, False, False, ..., False]
switching_exog_list = [True] + [False] * len(exog_cols)

# Instanciar o modelo
model_complex = sm.tsa.MarkovRegression(
    endog, 
    k_regimes=3,                  # 3 regimes (Calmo, Fuga, Euforia)
    exog=exog_with_const,         # Incluindo as 14 features + constante
    switching_variance=True,      # A volatilidade (sigma2) muda em cada regime
    switching_exog=switching_exog_list # APENAS o intercepto muda
)

# Treinar (fittar) o modelo
# Usamos 'search_reps' para encontrar um ponto de partida melhor e evitar
# mínimos locais, o que melhora a chance de convergência.
try:
    result_complex = model_complex.fit(maxiter=200, search_reps=10)
    
    print("\n--- 4. Modelo Treinado com Sucesso ---")
    
    # --- 4. Exibindo os Resultados ---
    print(result_complex.summary())

except Exception as e:
    print(f"\nErro ao treinar o Modelo: {e}")
    print("Isso pode ocorrer devido a problemas de convergência.")