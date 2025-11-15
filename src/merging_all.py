import pandas as pd
import numpy as np

# Carrega seus dados originais (Aave, SOFR, etc.)
df_original = pd.read_csv("../data/dataset_final.csv")
df_original = df_original.rename(columns={'Unnamed: 0' : 'date'})

# Carrega os dados do Compound que acabamos de tratar
df_compound = pd.read_csv('../raw_data/Compound/compound-final.csv')


# --- 2. Preparar para a União (Merge) ---

# Garantir que as colunas de data em AMBOS os arquivos
# sejam objetos datetime e virem o índice.

# ATENÇÃO: Verifique se a coluna de data no seu 'dataset_final.csv'
# se chama 'date'. Se tiver outro nome (ex: 'Date', 'timestamp'),
# ajuste a linha abaixo.
df_original['date'] = pd.to_datetime(df_original['date'])
df_original = df_original.set_index('date')

# Fazer o mesmo para o arquivo do compound
df_compound['date'] = pd.to_datetime(df_compound['date'])
df_compound = df_compound.set_index('date')


# --- 3. Unir (Merge) os Datasets ---

# 'how='inner'' garante que só vamos manter os dias
# onde temos dados TANTO do dataset original QUANTO do compound.
df_merged = pd.merge(df_original, df_compound, left_index=True, right_index=True, how='inner')

print(f"--- Dados Unificados ---")
print(f"Dimensão do dataset combinado: {df_merged.shape}")
print(df_merged.head())
print("\nVerificando dados faltantes após o merge (deve ser 0):")
print(df_merged.isna().sum())
print("-" * 30)


# --- 4. Engenharia de Features ---

print("\n--- Iniciando Engenharia de Features ---")

# ### ATENÇÃO: Suposição Crítica ###
# Para criar o APY ponderado, preciso da coluna de TVL (Total Deposit) do Aave.
# Estou assumindo que ela se chama 'aave_total_deposit_usd'.
# **Se o nome for outro no seu 'dataset_final.csv', por favor, altere a linha abaixo.**
COLUNA_AAVE_TVL = 'totalValueLockedUSD' # <--- AJUSTE AQUI SE NECESSÁRIO

# --- 4.1. Criando o Novo Alvo Ponderado (y) ---

# Renomear colunas para clareza
df_merged['total_tvl_aave'] = df_merged[COLUNA_AAVE_TVL]
df_merged['total_tvl_compound'] = df_merged['total_deposit_usd']
df_merged['total_tvl_defi'] = df_merged['total_tvl_aave'] + df_merged['total_tvl_compound']

# Renomear APYs para clareza
df_merged['apy_aave'] = df_merged['Y_Aave_APY'] # Do seu dataset original
df_merged['apy_compound'] = df_merged['deposit_apy'] # Do novo dataset do compound

# Calcular o APY Ponderado
df_merged['DeFi_APY_Ponderado'] = (
    (df_merged['apy_aave'] * df_merged['total_tvl_aave']) +
    (df_merged['apy_compound'] * df_merged['total_tvl_compound'])
) / df_merged['total_tvl_defi']

# Criar a nova variável alvo (Y)
df_merged['Y_Target_Spread_Ponderado'] = df_merged['DeFi_APY_Ponderado'] - df_merged['Y_SOFR']
print("Novo alvo 'Y_Target_Spread_Ponderado' criado.")

# --- 4.2. Criando Novas Features (para o X) ---

# 1. Spread Interno DeFi (Aave vs Compound)
df_merged['Spread_Aave_Compound'] = df_merged['apy_aave'] - df_merged['apy_compound']

# 2. Taxa de Utilização do Compound (já veio pronta, apenas pegamos)
df_merged['compound_utilization_rate'] = df_merged['utilization_rate']

# 3. Fluxo de Capital (Variação % de 7 dias do TVL DeFi total)
# .pct_change(7) calcula a variação percentual em 7 dias
df_merged['delta_total_defi_tvl_7d'] = df_merged['total_tvl_defi'].pct_change(7)
print("Novas features para clusterização criadas.")

# --- 5. Limpeza Final ---

# A feature 'delta_total_defi_tvl_7d' criou NaNs nas primeiras 7 linhas.
# Vamos removê-los para ter um dataset 100% limpo.
df_final = df_merged.dropna()

print(f"\nDimensão final do dataset após engenharia: {df_final.shape}")

# Salvar o dataset final e pronto para modelagem
df_final.to_csv('../data/dataset-final-final.csv')

print("\n--- Dataset Final (Pronto para Modelagem) ---")
print(df_final.head())