import requests
import pandas as pd
import yfinance as yf
import datetime
import matplotlib.pyplot as plt

# --- CONFIGURAÇÃO ---
POOLS = {
    "Lido_stETH": "747c1d2a-c668-4682-b9f9-296708a3dd90", 
    "Aave_V2_USDC": "aa70268e-4b52-42bf-a116-608b370f9501" 
}

START_DATE = "2022-01-01" 
END_DATE = datetime.datetime.today().strftime('%Y-%m-%d')

def get_defi_history(pool_id, pool_name):
    print(f"🔄 Buscando dados DeFi para: {pool_name}...")
    url = f"https://yields.llama.fi/chart/{pool_id}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()['data']
        
        if not data:
            print(f"⚠️ Dados vazios para {pool_name}")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        
        # 1. TRATAMENTO DE DATA
        df['date'] = pd.to_datetime(df['timestamp'], utc=True)
        df['date'] = df['date'].dt.tz_localize(None)
        df['date'] = df['date'].dt.normalize() 
        
        # 2. TRATAMENTO DE NÚMEROS
        # Convertemos APY e TVL para números antes de qualquer filtro
        df['apy'] = pd.to_numeric(df['apy'], errors='coerce')
        
        # Verifica se TVL existe e converte
        if 'tvlUsd' in df.columns:
            df['tvlUsd'] = pd.to_numeric(df['tvlUsd'], errors='coerce')
        
        # Remove linhas onde o APY é inválido
        df = df.dropna(subset=['apy'])

        # Define o índice
        df = df.set_index('date')
        
        # 3. FILTRAGEM E RENOMEAÇÃO (Antes de agrupar)
        cols_to_keep = ['apy', 'tvlUsd']
        existing_cols = [c for c in cols_to_keep if c in df.columns]
        df = df[existing_cols]
        
        # Renomeia as colunas para o formato final
        rename_map = {
            'apy': f'{pool_name}_APY',
            'tvlUsd': f'{pool_name}_TVL'
        }
        df = df.rename(columns=rename_map)
        
        # 4. AGREGAÇÃO FINAL
        # Calcula a média do dia para TODAS as colunas numéricas que sobraram (APY e TVL)
        df = df.groupby(df.index).mean()

        print(f"✅ {pool_name}: {len(df)} registros válidos.")
        return df

    except Exception as e:
        print(f"❌ Erro em {pool_name}: {e}")
        return pd.DataFrame()

def get_tradfi_history(tickers):
    print(f"🔄 Buscando dados TradFi (Yahoo Finance)...")
    # Baixa dados
    df = yf.download(tickers, start=START_DATE, end=END_DATE, progress=False)['Close']
    
    # Tratamento de índice de data
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    df.index = df.index.normalize()
    
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.droplevel(0)
        except:
            pass
            
    mapper = {'^TNX': 'US_Treasury_10Y', '^VIX': 'VIX_Index'}
    df = df.rename(columns=mapper)
    print(f"✅ TradFi: {len(df)} registros encontrados.")
    return df

# --- EXECUÇÃO ---

# 1. Coleta
df_lido = get_defi_history(POOLS["Lido_stETH"], "Lido")
df_lido.to_csv("dataset_lido.csv") # Backup

df_aave = get_defi_history(POOLS["Aave_V2_USDC"], "Aave_USDC")
df_aave.to_csv("dataset_aave.csv") # Backup

df_tradfi = get_tradfi_history(['^TNX', '^VIX'])
df_tradfi.to_csv("dataset_tradfi.csv") # Backup

print("\n--- CONSOLIDAÇÃO ---")

if df_lido.empty or df_aave.empty:
    print("❌ ERRO CRÍTICO: Um dos datasets DeFi está vazio.")
else:
    # Merge
    df_final = df_lido.join(df_aave, how='outer').join(df_tradfi, how='outer')

    # Filtro de Data e Preenchimento
    df_final = df_final[df_final.index >= pd.to_datetime(START_DATE)]
    df_final = df_final.ffill()
    df_final = df_final.bfill()
    df_final = df_final.dropna()

    # Feature Engineering
    if 'Aave_USDC_APY' in df_final.columns and 'US_Treasury_10Y' in df_final.columns:
        df_final['Spread_Aave_Treasury'] = df_final['Aave_USDC_APY'] - df_final['US_Treasury_10Y']
        df_final['Spread_Staking_Risk'] = df_final['Lido_APY'] - df_final['US_Treasury_10Y']

        print(f"\n📊 SUCESSO! Dataset Final: {len(df_final)} linhas.")
        print(df_final.tail()) # Veja se aparecem as colunas _TVL aqui

        filename = 'dataset_datathon_final.csv'
        df_final.to_csv(filename)
        print(f"\n💾 Arquivo salvo: {filename}")
        
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(df_final.index, df_final['Spread_Aave_Treasury'], label='Spread Aave')
            plt.axhline(0, color='red', linestyle='--')
            plt.legend()
            plt.savefig("chart_debug.png")
            print("📈 Gráfico salvo.")
        except:
            pass

    else:
        print("❌ Erro: Colunas necessárias não foram criadas no Merge.")