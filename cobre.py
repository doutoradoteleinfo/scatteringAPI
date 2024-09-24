import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

# Importando Dataset
df = pd.read_csv('/Users/MacBarroso/PycharmProjects/API/dataset_COBRE.csv', sep=';')

# Ajustar o valor de lambda (multiplicando por 1000 para manter a escala compatível)
df['lambda'] = df['lambda'] * 1000

# Separar as variáveis independentes e dependente
modelo_features = ["altura", "raio", "lambda"]
X = df[modelo_features]
y = df.scattering

# Criar um dicionário para armazenar os modelos treinados
model_dict = {}

# Dividir o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

# Treinar o modelo de Random Forest
rfr = RandomForestRegressor(n_estimators=100, random_state=0)
rfr.fit(X_train, y_train)

# Armazenar o modelo no dicionário com a chave correta (em minúsculas)
model_dict['copper_model'] = rfr

# Salvar o dicionário com o modelo em um arquivo .pkl
with open('COBRE_model.pkl', 'wb') as file:
    pickle.dump(model_dict, file)
