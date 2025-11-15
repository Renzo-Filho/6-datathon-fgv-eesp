import pandas as pd
from model import Model

# ---------------------------------------------------------------
# 1. INICIALIZAÇÃO E TREINAMENTO
# ---------------------------------------------------------------
# Ao criar o objeto, ele já carrega os dados e chama _train()
print("="*50)
print("ETAPA 1: Inicializando e Treinando o Modelo...")
print("="*50)
# Podemos ajustar os quantis de threshold aqui, se quisermos
model = Model(volatility_threshold=0.7, utilization_threshold=0.7)


# ---------------------------------------------------------------
# 2. AVALIAÇÃO DO MODELO
# ---------------------------------------------------------------
# O método evaluate() usa o X_test interno do modelo para gerar métricas
print("\n" + "="*50)
print("ETAPA 2: Avaliando o Modelo no Conjunto de Teste...")
print("="*50)
metrics, test_predictions = model.evaluate()


# ---------------------------------------------------------------
# 3. VISUALIZAÇÃO DOS REGIMES
# ---------------------------------------------------------------
# O método view_clusters_plot() usa o df_plot interno
print("\n" + "="*50)
print("ETAPA 3: Gerando Gráfico de Regimes...")
print("="*50)
model.view_clusters_plot()


# ---------------------------------------------------------------
# 4. EXEMPLO DE PREVISÃO EM NOVOS DADOS
# ---------------------------------------------------------------
# Vamos simular a chegada de 5 novas amostras de dados
# (Pegando as 5 primeiras linhas do conjunto de teste como exemplo)
print("\n" + "="*50)
print("ETAPA 4: Exemplo de Previsão em Novos Dados...")
print("="*50)

# Pegamos os dados de input (features)
X_new_sample = model.X_test.iloc[0:5]

# Pegamos os dados reais (para comparar)
Y_actual_sample = model.Y_test.iloc[0:5]

# Fazemos a previsão
new_predictions = model.predict(X_new_sample)

# Exibir os resultados
print("\nPrevisões para as 5 novas amostras:")
results_df = pd.DataFrame({
    'Spread_Real': Y_actual_sample,
    'Spread_Previsto': new_predictions
})
print(results_df)
print("\nExemplo completo concluído.")