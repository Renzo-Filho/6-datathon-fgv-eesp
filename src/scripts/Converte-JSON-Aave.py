import pandas as pd
import json

# Exibir configuração para mais colunas
pd.set_option('display.max_columns', None)

# --- 1. Carregar e Juntar os Ficheiros JSON ---

try:
    # Carregar o primeiro ficheiro (com os 1000 resultados)
    with open('json.json', 'r') as f:
        data_p1 = json.load(f)

    # Carregar o segundo ficheiro (com os 22 resultados)
    with open('json2.json', 'r') as f:
        data_p2 = json.load(f)

    # Extrair a lista de snapshots de cada página
    snapshots_p1 = data_p1.get('data', {}).get('marketDailySnapshots', [])
    snapshots_p2 = data_p2.get('data', {}).get('marketDailySnapshots', [])
    
    # Juntar as duas listas
    lista_completa = snapshots_p1 + snapshots_p2
    
    total_snapshots = len(lista_completa)
    print(f"Total de snapshots na Página 1: {len(snapshots_p1)}")
    print(f"Total de snapshots na Página 2: {len(snapshots_p2)}")
    print(f"Total de snapshots combinados: {total_snapshots}")

    if total_snapshots == 0:
        print("Erro: Nenhum dado encontrado nos ficheiros JSON. Verifique os ficheiros.")
    else:
        # --- 2. Criar o DataFrame Mestre ---
        df_mestre = pd.DataFrame(lista_completa)

        # --- 3. Achatamento (Flattening) das colunas-objeto ---
        
        # APY de Depósito (Supply)
        df_mestre['Y_Aave_APY'] = df_mestre['supplyAPY'].apply(lambda x: float(x[0]['rate']) if (x and isinstance(x, list) and len(x) > 0) else None)
        
        # APY de Empréstimo Variável
        df_mestre['X_Variable_Borrow_APY'] = df_mestre['borrowAPY'].apply(lambda x: float(x[0]['rate']) if (x and isinstance(x, list) and len(x) > 0) else None)
        
        # APY de Empréstimo Estável
        df_mestre['X_Stable_Borrow_APY'] = df_mestre['stableBorrowAPY'].apply(lambda x: float(x[0]['rate']) if (x and isinstance(x, list) and len(x) > 0) else None)
        
        # Símbolo do Token
        df_mestre['token_symbol'] = df_mestre['market'].apply(lambda x: x.get('inputToken', {}).get('symbol') if (x and isinstance(x, dict)) else None)

        # --- 4. Conversão de Tipos de Dados ---

        # Converter colunas USD para numérico
        usd_cols = ['totalBorrowBalanceUSD', 'totalDepositBalanceUSD', 'totalValueLockedUSD', 'cumulativeSupplySideRevenueUSD']
        for col in usd_cols:
            df_mestre[col] = pd.to_numeric(df_mestre[col], errors='coerce')

        # Converter Timestamp para Datetime
        df_mestre['timestamp'] = pd.to_numeric(df_mestre['timestamp'])
        df_mestre['date'] = pd.to_datetime(df_mestre['timestamp'], unit='s')

        # --- 5. Cálculo de Features (como no seu PDF) ---

        # Calcular a Taxa de Utilização (X_Aave_Utilization)
        df_mestre['X_Aave_Utilization'] = df_mestre.apply(
            lambda row: row['totalBorrowBalanceUSD'] / row['totalDepositBalanceUSD'] if pd.notnull(row['totalDepositBalanceUSD']) and row['totalDepositBalanceUSD'] > 0 else 0,
            axis=1
        )

        # --- 6. Selecionar Colunas e Exportar ---

        colunas_finais = [
            'date',
            'timestamp',
            'token_symbol',
            'Y_Aave_APY',
            'X_Variable_Borrow_APY',
            'X_Stable_Borrow_APY',
            'X_Aave_Utilization',
            'totalBorrowBalanceUSD',
            'totalDepositBalanceUSD',
            'totalValueLockedUSD',
            'cumulativeSupplySideRevenueUSD',
            'blockNumber'
        ]
        
        # Garantir que apenas as colunas que existem são selecionadas
        colunas_existentes = [col for col in colunas_finais if col in df_mestre.columns]
        df_final_limpo = df_mestre[colunas_existentes]

        # Ordenar por data (mais antigo primeiro, para séries temporais)
        df_final_limpo = df_final_limpo.sort_values(by='date', ascending=True).reset_index(drop=True)

        # --- 7. Exportar para CSV ---
        nome_do_ficheiro_csv = 'aave_usdc_dados_completos.csv'
        df_final_limpo.to_csv(nome_do_ficheiro_csv, index=False, encoding='utf-8')

        print(f"\nDataFrame completo ({total_snapshots} linhas) exportado com sucesso para: {nome_do_ficheiro_csv}")
        print("\nVisualizando as primeiras 5 linhas do DataFrame final (dados mais antigos):")
        display(df_final_limpo.head())
        
        print("\nVisualizando as últimas 5 linhas do DataFrame final (dados mais recentes):")
        display(df_final_limpo.tail())
        
        print(f"\nInformações do DataFrame final:")
        df_final_limpo.info()

except FileNotFoundError as e:
    print(f"Erro: Ficheiro não encontrado. Certifique-se que 'json.json' e 'json2.json' estão no diretório. Detalhes: {e}")
except Exception as e:
    print(f"Ocorreu um erro inesperado durante o processamento: {e}")
    import traceback
    traceback.print_exc()
