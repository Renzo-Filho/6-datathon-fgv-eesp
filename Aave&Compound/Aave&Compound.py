import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def main():
    # --- 1. Carregar Dados ---
    try:
        df_compound = pd.read_csv("compound_v3_data.csv")
        df_aave = pd.read_csv("dataset_final.csv")
        print("Arquivos CSV carregados com sucesso.")
    except FileNotFoundError as e:
        print(f"Erro: Arquivo não encontrado. {e}")
        print("Por favor, certifique-se que 'compound_v3_data.csv' e 'dataset_final.csv' estão na mesma pasta que o script.")
        return # Sai do script se os arquivos não forem encontrados

    # --- 2. Processar Dados do Compound ---
    # Verificar se as colunas necessárias existem
    if 'date_time' not in df_compound.columns or 'deposit_apy' not in df_compound.columns:
        print("Erro: Colunas 'date_time' ou 'deposit_apy' não encontradas no arquivo Compound.")
        return

    df_compound_proc = df_compound[['date_time', 'deposit_apy']].copy()
    # Converter para datetime
    df_compound_proc['date_time'] = pd.to_datetime(df_compound_proc['date_time'])
    # Remover NaNs no APY
    df_compound_proc = df_compound_proc.dropna(subset=['deposit_apy'])
    
    # Agregar por dia (média diária), pois Aave é diário
    df_compound_daily = df_compound_proc.set_index('date_time').resample('D').mean()
    df_compound_daily.rename(columns={'deposit_apy': 'Compound_APY'}, inplace=True)
    print("Dados do Compound processados (agregados por dia).")

    # --- 3. Processar Dados da Aave ---
    if 'Unnamed: 0' not in df_aave.columns or 'Y_Aave_APY' not in df_aave.columns:
        print("Erro: Colunas 'Unnamed: 0' (data) ou 'Y_Aave_APY' não encontradas no arquivo Aave.")
        return
        
    df_aave_proc = df_aave[['Unnamed: 0', 'Y_Aave_APY']].copy()
    df_aave_proc.rename(columns={'Unnamed: 0': 'date', 'Y_Aave_APY': 'Aave_APY'}, inplace=True)
    # Converter para datetime
    df_aave_proc['date'] = pd.to_datetime(df_aave_proc['date'])
    # Definir data como índice
    df_aave_daily = df_aave_proc.set_index('date')
    print("Dados da Aave processados.")

    # --- 4. Combinar os Dados ---
    # Usar 'inner' join para garantir que tenhamos dados de ambos para as datas plotadas
    df_combined = pd.concat([df_compound_daily, df_aave_daily], axis=1, join='inner')

    if df_combined.empty:
        print("Erro: Após a combinação, não há dados correspondentes entre as datas dos dois arquivos.")
        return
        
    print(f"Dados combinados, resultando em {len(df_combined)} pontos de dados em comum.")

    # --- 5. Gerar o Gráfico com Matplotlib ---
    plt.figure(figsize=(12, 6))

    # Plotar as duas séries
    plt.plot(df_combined.index, df_combined['Compound_APY'], label='Compound APY', color='blue')
    plt.plot(df_combined.index, df_combined['Aave_APY'], label='Aave APY', color='orange')

    # Configurar Título e Eixos
    plt.title('Comparação APY: Compound vs Aave', fontsize=16)
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('APY (%)', fontsize=12)

    # Adicionar Legenda
    plt.legend()

    # Adicionar Grid
    plt.grid(True, linestyle='--', alpha=0.6)

    # Melhorar a formatação do eixo X (Datas)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate() # Auto-rotacionar datas

    # Ajustar layout
    plt.tight_layout()

    # Salvar a imagem
    plt.savefig('compound_vs_aave_apy_matplotlib.png')
    plt.show()
    print("Gráfico do Matplotlib salvo como 'compound_vs_aave_apy_matplotlib.png'")

    # Opcional: Mostrar o gráfico (descomente se quiser que ele apareça ao rodar o script)
    # plt.show()

# Executar a função principal
if __name__ == "__main__":
    main()