import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

# Assumindo que 'X_y_df' e 'target_col' existem do Bloco 1

try:
    print("\nIniciando Treinamento (Teste 15 - GARCH Volatilidade)...")

    # GARCH funciona melhor em "retornos" (deltas), não em "níveis"
    y_spread_deltas = X_y_df[target_col].diff(1).dropna()
    
    # GARCH(1,1) é o modelo padrão
    # p=1 (lag de volatilidade), q=1 (lag de erro quadrado)
    garch_model = arch_model(y_spread_deltas, vol='Garch', p=1, q=1)
    
    # Treinar (fit) o modelo
    garch_results = garch_model.fit(disp='off')
    
    print(garch_results.summary())
    
    # --- Plotar a Volatilidade (Risco) ---
    print("\nPlotando a volatilidade condicional (o risco previsto)...")
    
    # Pegar a volatilidade prevista pelo modelo
    conditional_volatility = garch_results.conditional_volatility
    
    plt.figure(figsize=(15, 6))
    plt.plot(df_clean['date'].iloc[1:], y_spread_deltas, label='Mudança Diária no Spread (Deltas)', alpha=0.7)
    plt.plot(
        df_clean['date'].iloc[1:], 
        conditional_volatility, 
        label='Volatilidade GARCH (Risco)', 
        color='red', 
        linestyle='--'
    )
    plt.title('Modelo GARCH: Volatilidade do Spread de Risco DeFi')
    plt.legend()
    plt.show()
    
    # Salvar o gráfico
    plt.savefig('garch_volatility_plot.png')
    print("Gráfico GARCH salvo em 'garch_volatility_plot.png'")

except Exception as e:
    print(f"Um erro ocorreu durante o treinamento do GARCH: {e}")