import pandas as pd
import numpy as np

# Nomes dos arquivos
file_btc = "../raw_data/features/CBBTCUSD.csv"
file_eth = "../raw_data/features/CBETHUSD.csv"
file_dgs2 = "../raw_data/features/DGS2.csv"
file_gas = "../raw_data/features/export-AvgGasPrice.csv"
file_vix = "../raw_data/features/VIXCLS.csv"
file_aave_base = "../raw_data/aave/aave_cleaned.csv" # Nosso DataFrame base (já inspecionado)
file_sofr_base = "../raw_data/sofr/SOFR90DAYAVG.csv" # Nosso Y2 (já inspecionado)

# --- 1. Preparar DataFrame Base (Aave - Y1 e X-on-chain) ---
df_aave = pd.read_csv(file_aave_base)
df_aave['date'] = pd.to_datetime(df_aave['date']).dt.tz_localize(None) # Remove fuso horário
df_aave = df_aave.set_index('date')
# Agregar por dia (resample) e manter apenas colunas numéricas
df_base = df_aave.resample('D').mean(numeric_only=True)
# Renomear colunas para clareza
df_base = df_base.rename(columns={
    'aave_usdc_apy': 'Y1_Aave_APY',
    'aave_utilization_rate': 'X_Aave_Utilization',
    'usdc_depeg_risk': 'X_USDC_Depeg_Risk'
})
# Selecionar apenas as colunas que importam do Aave
df_base = df_base[['Y1_Aave_APY', 'X_Aave_Utilization', 'X_USDC_Depeg_Risk']]


# --- 2. Preparar DataFrame TradFi (Y2 e X-off-chain) ---
# Carregar dados
df_sofr = pd.read_csv(file_sofr_base)
df_vix = pd.read_csv(file_vix)
df_dgs2 = pd.read_csv(file_dgs2)

# Renomear colunas antes do merge
df_sofr = df_sofr.rename(columns={'SOFR90DAYAVG': 'Y2_Ondo_Proxy_APY', 'observation_date': 'date'})
df_vix = df_vix.rename(columns={'VIXCLS': 'X_VIX', 'observation_date': 'date'})
df_dgs2 = df_dgs2.rename(columns={'DGS2': 'X_DGS2', 'observation_date': 'date'})

# Converter datas para datetime
df_sofr['date'] = pd.to_datetime(df_sofr['date'])
df_vix['date'] = pd.to_datetime(df_vix['date'])
df_dgs2['date'] = pd.to_datetime(df_dgs2['date'])

# Juntar todos os dados TradFi
df_tradfi = pd.merge(df_sofr, df_vix, on='date', how='outer')
df_tradfi = pd.merge(df_tradfi, df_dgs2, on='date', how='outer')

# Definir data como índice e aplicar ffill()
df_tradfi = df_tradfi.set_index('date').sort_index()
# REGRA DO ROTEIRO: Aplicar ffill() para preencher fins de semana
df_tradfi = df_tradfi.resample('D').ffill()


# --- 3. Preparar DataFrame Crypto (X-on-chain e X-off-chain) ---
# Carregar dados
df_btc = pd.read_csv(file_btc)
df_eth = pd.read_csv(file_eth)
df_gas = pd.read_csv(file_gas, quotechar='"', thousands=',')

# Limpar Gas
df_gas.columns = df_gas.columns.str.replace('"', '').str.strip() # Limpa nomes das colunas
df_gas['Value (Wei)'] = pd.to_numeric(df_gas['Value (Wei)'], errors='coerce')
df_gas['X_Gas_Price_Gwei'] = df_gas['Value (Wei)'] / 1e9 # Converter Wei para Gwei
df_gas = df_gas.rename(columns={'Date(UTC)': 'date'})
df_gas['date'] = pd.to_datetime(df_gas['date'])
df_gas = df_gas.set_index('date').resample('D').mean(numeric_only=True)

# Limpar BTC
df_btc = df_btc.rename(columns={'CBBTCUSD': 'X_BTC_Price', 'observation_date': 'date'})
df_btc['date'] = pd.to_datetime(df_btc['date'])
df_btc = df_btc.set_index('date').resample('D').mean(numeric_only=True)

# Limpar ETH
df_eth = df_eth.rename(columns={'CBETHUSD': 'X_ETH_Price', 'observation_date': 'date'})
df_eth['date'] = pd.to_datetime(df_eth['date'])
df_eth = df_eth.set_index('date').resample('D').mean(numeric_only=True)

# Juntar dados Crypto
df_crypto = df_btc.join(df_eth, how='outer')
df_crypto = df_crypto.join(df_gas, how='outer')


# --- 4. Merge Final ---
# Juntar os 3 blocos. Usamos 'inner' para manter apenas dias com dados de todas as fontes.
df_master = df_base.join(df_tradfi, how='inner')
df_master = df_master.join(df_crypto, how='inner')


# --- 5. Engenharia de Features (O "Ouro" da Fase 2) ---
print(f"Formato antes da Eng. Features: {df_master.shape}")

# 1. Variável Alvo ($Y_target$)
df_master['Y_Target_Spread'] = df_master['Y1_Aave_APY'] - df_master['Y2_Ondo_Proxy_APY']

# 2. Features do Roteiro
df_master['X_ETH_Vol_7D'] = df_master['X_ETH_Price'].pct_change().rolling(window=7).std() * np.sqrt(7) # Volatilidade anualizada (opcional, mas std simples é ok)
df_master['X_Gas_MA_7D'] = df_master['X_Gas_Price_Gwei'].rolling(window=7).mean()
df_master['X_VIX_Pct_Change_1D'] = df_master['X_VIX'].pct_change(1)
df_master['X_Aave_Util_Accel_1D'] = df_master['X_Aave_Utilization'].diff(1)


# --- 6. Limpeza Final ---
# Remover NaNs criados pelas médias móveis (rolling) e diffs
df_master = df_master.dropna()

# --- 7. Inspecionar e Salvar ---
print(f"Formato após Eng. Features e dropna: {df_master.shape}")

print("\n--- Inspecionando o DataFrame Master Final (head) ---")
print(df_master.head())

print("\n--- Inspecionando o DataFrame Master Final (info) ---")
df_master.info()

# Salvar o dataset final
df_master.to_csv("../data/dataset.csv")