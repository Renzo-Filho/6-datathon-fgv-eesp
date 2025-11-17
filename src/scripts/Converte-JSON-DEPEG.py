import pandas as pd
import json
from decimal import Decimal, getcontext

# Definir a precisão para os cálculos decimais
getcontext().prec = 50

def process_snapshots(snapshot_list):
    """
    Processa uma lista de snapshots, normalizando balanços e calculando o risco.
    """
    processed_data = []
    
    # NOTA: Assumindo a ordem [DAI, USDC, USDT]
    # DAI (Índice 0) -> 18 casas decimais (1e18)
    # USDC (Índice 1) -> 6 casas decimais (1e6)
    # USDT (Índice 2) -> 18 casas decimais (1e18)
    
    DECIMAL_DAI = Decimal('1e18')
    DECIMAL_USDC = Decimal('1e6')
    DECIMAL_USDT = Decimal('1e6')

    for snapshot in snapshot_list:
        try:
            timestamp = int(snapshot['timestamp'])
            balances = snapshot['inputTokenBalances']

            # Converter os balanços brutos (strings) para Decimal
            raw_dai = Decimal(balances[0])
            raw_usdc = Decimal(balances[1])
            raw_usdt = Decimal(balances[2])

            # Normalizar para o valor em "dólar" (removendo as casas decimais)
            norm_dai = raw_dai / DECIMAL_DAI
            norm_usdc = raw_usdc / DECIMAL_USDC
            norm_usdt = raw_usdt / DECIMAL_USDT

            # Calcular o balanço total normalizado
            norm_total = norm_dai + norm_usdc + norm_usdt

            # Calcular a feature de risco
            if norm_total == 0:
                depeg_risk = Decimal(0)
            else:
                # Risco = (Saldos de USDC) / (Saldos Totais)
                depeg_risk = norm_usdc / norm_total

            processed_data.append({
                'timestamp': timestamp,
                'date': pd.to_datetime(timestamp, unit='s'),
                'X_USDC_Depeg_Risk': depeg_risk,
                'balance_usdc_normalized': norm_usdc,
                'balance_dai_normalized': norm_dai,
                'balance_usdt_normalized': norm_usdt,
                'balance_total_normalized': norm_total,
                'raw_balance_usdc': balances[1],
                'raw_balance_dai': balances[0],
                'raw_balance_usdt': balances[2],
            })
        except Exception as e:
            print(f"Erro ao processar o snapshot {snapshot}: {e}")
            
    return processed_data

# --- Script Principal ---
def main():
    files_to_process = ['json1.1.json', 'json1.2.json']
    all_snapshots = []

    # 1. Carregar e combinar dados de ambos os arquivos
    for filename in files_to_process:
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                snapshots = data['data']['liquidityPoolDailySnapshots']
                all_snapshots.extend(snapshots)
                print(f"Carregados {len(snapshots)} snapshots de {filename}")
        except FileNotFoundError:
            print(f"ERRO: Arquivo '{filename}' não encontrado.")
            print("Certifique-se de que os arquivos JSON estão na mesma pasta que o script.")
            return
        except Exception as e:
            print(f"Erro ao ler {filename}: {e}")
            return

    print(f"Total de snapshots para processar: {len(all_snapshots)}")

    # 2. Processar a lista combinada
    processed_list = process_snapshots(all_snapshots)

    if not processed_list:
        print("Nenhum dado processado.")
        return

    # 3. Criar o DataFrame
    df = pd.DataFrame(processed_list)

    # 4. Ordenar por data (timestamp) para criar a série temporal correta
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    # 5. Definir a ordem final das colunas para o CSV
    columns_order = [
        'date', 
        'timestamp', 
        'X_USDC_Depeg_Risk', 
        'balance_usdc_normalized',
        'balance_dai_normalized',
        'balance_usdt_normalized',
        'balance_total_normalized',
        'raw_balance_usdc',
        'raw_balance_dai',
        'raw_balance_usdt'
    ]
    df = df[columns_order]

    # 6. Salvar em CSV
    output_filename = 'curve_3pool_usdc_depeg_risk_completo.csv'
    df.to_csv(output_filename, index=False, float_format='%.18f') # Salva com alta precisão

    print(f"\nSucesso! 🚀")
    print(f"Arquivo CSV completo salvo como: {output_filename}")
    
    print("\n--- 5 Primeiras Linhas do CSV ---")
    print(df.head().to_string())

if __name__ == "__main__":
    main()
