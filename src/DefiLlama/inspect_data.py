import requests
import pandas as pd

# ID do Aave V2 USDC (O mesmo que usamos)
POOL_ID = "aa70268e-4b52-42bf-a116-608b370f9501" 

url = f"https://yields.llama.fi/chart/{POOL_ID}"
response = requests.get(url)
data = response.json()['data']

if data:
    # Pega o último registro (dia mais recente) para ver as chaves
    latest_record = data[-1]
    print("--- DADOS DISPONÍVEIS NESTA POOL ---")
    for key, value in latest_record.items():
        print(f"🔑 {key}: {value}")
    
    # Cria um DataFrame rápido para ver as colunas
    df = pd.DataFrame(data)
    print("\n--- VISÃO TABULAR (Primeiras 5 linhas) ---")
    print(df.head())
    print("\n--- COLUNAS DISPONÍVEIS ---")
    print(df.columns.tolist())
else:
    print("Nenhum dado encontrado.")