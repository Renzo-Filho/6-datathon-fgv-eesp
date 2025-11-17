import pandas as pd

# --- 1. Carregar os Dados do Compound ---
file_path = '../raw_data/Compound/compound_v3_data.csv'
df_compound = pd.read_csv(file_path)

print(f"Dados originais carregados: {df_compound.shape[0]} linhas.")

# --- 2. Limpeza e Seleção de Colunas ---
# Vamos manter apenas as colunas que interessam para a análise
colunas_uteis = [
    'date_time', 
    'deposit_apy', 
    'borrow_apy', 
    'utilization_rate', 
    'total_deposit_usd', 
    'total_borrow_usd'
] # [cite: 1]
df_compound = df_compound[colunas_uteis]

# --- 3. Preparação da Coluna de Data (Obrigatório) ---
# Converter a coluna 'date_time' para o formato datetime do pandas
df_compound['date'] = pd.to_datetime(df_compound['date_time'])

# Para fazer a reamostragem, o índice DEVE ser a coluna de data
df_compound = df_compound.set_index('date')

# --- 4. Reamostragem (Resampling) para Frequência Diária ---
# 'D' = Diário (Daily).
# .mean() = Calcula a média de todos os pontos daquele dia.
# Se só houver um ponto (o que parece ser o caso), ele apenas usa aquele valor.
# numeric_only=True ignora colunas não numéricas (como a 'date_time' original)
df_compound_daily = df_compound.resample('D').mean(numeric_only=True)

print(f"Dados reamostrados para diário: {df_compound_daily.shape[0]} linhas.")

# --- 5. Tratamento de Dados Faltantes (NaNs) ---
# A reamostragem criará dias "em branco" (NaN) onde não havia dados.
# 'ffill' (forward fill) preenche o NaN com o último valor válido.
# Isso é uma premissa comum: o APY de ontem é o mesmo de hoje se não houver nova atualização.
df_compound_daily = df_compound_daily.fillna(method='ffill')

# O 'ffill' pode não preencher os primeiros dias se eles forem NaN.
# Vamos usar 'bfill' (backfill) para preencher o início com o primeiro dado válido.
df_compound_daily = df_compound_daily.fillna(method='bfill')

# --- 6. Resultado ---
print("\n--- Dados do Compound (Reamostrados e Prontos) ---")
print(df_compound_daily.head())

print("\n--- Verificando se ainda existem NaNs ---")
# O resultado aqui deve ser 0 para todas as colunas
print(df_compound_daily.isna().sum())

df_compound_daily.to_csv("../raw_data/Compound/compound-final.csv")
