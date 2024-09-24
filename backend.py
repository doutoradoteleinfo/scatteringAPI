import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

# Importando Dataset
df = pd.read_csv('/Users/MacBarroso/PycharmProjects/API/dados_1.CSV', sep=';')

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
model_dict['gold_model'] = rfr

# Salvar o dicionário com o modelo em um arquivo .pkl
with open('OURO_model.pkl', 'wb') as file:
    pickle.dump(model_dict, file)




# #Importando bibliotecas necessárias
# import pandas as pd
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# import pickle
#
# #Importando Dataset
# df = pd.read_csv('/Users/MacBarroso/PycharmProjects/API/dados_1.CSV',sep=';')
#
# # print(data_regression.shape)
#
# # data_1 = data_regression[data_regression['altura'] == 100.0]
# # data_test = data_1[(data_1['raio'] == 60.0) | (data_1['raio'] == 150.0)]
# # data_test_h100r150 = data_test[(data_test['altura'] == 100.0) & (data_test['raio'] == 150.0)]
# # data_test_h100r60 = data_test[(data_test['altura'] == 100.0) & (data_test['raio'] == 60.0)]
# # data_2 = data_regression[data_regression['altura'] == 150.0]
# # data_test_h150r70 = data_2[(data_2['altura'] == 150.0) & (data_2['raio'] == 70.0)]
#
# # data_train = data_regression[~data_regression.index.isin(data_test_h100r150.index) & ~data_regression.index.isin(data_test_h100r60.index) & ~data_regression.index.isin(data_test_h150r70.index)]
#
# # data_train = data_regression
# # print(data_train.shape)
#
# # Separar as variáveis independentes e dependente
# modelo_features=["altura", "raio", "lambda"]
# X = df[modelo_features]
#
# y = df.scattering
#
# # Criar um dicionário para armazenar os modelos treinados
# model_dict = {}
#
# # #Definindo os parâmetros de treinamento
# # reg = setup(data_train, train_size = 0.7, session_id=2632)
# #
# # best = compare_models()
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
#
# rfr = RandomForestRegressor(n_estimators=100, random_state=0)
#
# rfr.fit(X_train, y_train)
#
# # # Obtendo as métricas e resultados de cada modelo
# # model_results = pull()
# # model_results.to_csv('/Users/MacBarroso/PycharmProjects/pythonProject/resultado_geral.csv', index=False)
#
# #Criando o modelo com melhores métricas
# # rf = create_model('rf')
# #
# # tuned_rf = tune_model(rf, optimize='mse', n_iter=3, fold=3)
# #
# # results = evaluate_model(tuned_rf)
#
# # #Finalizando o modelo ajustado
# # final_rf = finalize_model(tuned_rf)
#
# # Armazenar o modelo em model_dict
# model_dict['OURO_model'] = rfr
#
# # Salvar o dicionário com o modelo
# with open('OURO_model.pkl', 'wb') as file:
#     pickle.dump(model_dict, file)
#
#
# # #Matriz de Confusão
# #
# #
# # #Salvando o modelo
# # save_model(final_rf, 'Final ET Model 31Mar2023')
# #
# # #Carregando o modelo salvo
# # final_rf = finalize_model(tuned_rf)
# #
# # saved_final_rf = load_model('Final ET Model 31Mar2023')
# # #
# # # #
# # # #
# # # #
# # """Teste de Validaçao"""
# #
# #
# #
# #
# # new_prediction_h100r150 = predict_model(saved_final_rf, data=data_test_h100r150)
# #
# # # Encontrando o valor máximo da predição e dos dados reais
# # max_pred = np.max(new_prediction_h100r150['prediction_label'])
# # max_real = np.max(new_prediction_h100r150['scattering'])
# #
# # # Calculando o erro entre o valor máximo da predição e o valor máximo dos dados reais
# # error = np.abs((max_pred - max_real)/max_real)
# #
# # # Plotando o gráfico dos valores de predição
# # plt.plot(new_prediction_h100r150['lambda'], new_prediction_h100r150['scattering'], color='red')
# # plt.plot(new_prediction_h100r150['lambda'], new_prediction_h100r150['prediction_label'], linestyle='--', color='blue')
# #
# # # Adicionando uma legenda com os valores do pico de scattering e o erro
# # plt.text(0.05, 0.95, f'Error at Max Value: {error:.4f}', ha='left', va='top', transform=plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5'))
# # plt.text(0.05, 0.85, f'Peak Value (Prediction): {max_pred:.4f}', ha='left', va='top', transform=plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5'))
# # plt.text(0.05, 0.75, f'Peak Value (Label): {max_real:.4f}', ha='left', va='top', transform=plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5'))
# #
# # plt.title('Random Forest Regressor Predict [h100r150]')
# # plt.xlabel('Wavelength [nm]')
# # plt.ylabel('Scattering Cross Section x10^-14 [a.u]')
# # plt.show()
# # # #
# # # #
# # # #
# # # Criar dataframe com os valores plotados
# # df_plot = pd.DataFrame({
# #     'lambda': new_prediction_h100r150['lambda'],
# #     'scattering': new_prediction_h100r150['scattering'],
# #     'prediction_label': new_prediction_h100r150['prediction_label']
# # })
# #
# # # Salvar dataframe em um arquivo CSV
# # df_plot.to_csv('h100r150.csv', index=False)
# # #
# # #
# # #
# # # # #
# # # # #
# # """Teste de Validaçao"""
# # # #
# # # #
# # # #
# # # #
# # new_prediction_h100r60 = predict_model(saved_final_rf, data=data_test_h100r60)
# # #
# # #
# # # Encontrando o valor máximo da predição e dos dados reais
# # max_pred = np.max(new_prediction_h100r60['prediction_label'])
# # max_real = np.max(new_prediction_h100r60['scattering'])
# #
# # # Calculando o erro entre o valor máximo da predição e o valor máximo dos dados reais
# # error = np.abs((max_pred - max_real)/max_real)
# #
# # # Plotando o gráfico dos valores de predição
# # plt.plot(new_prediction_h100r60['lambda'], new_prediction_h100r60['scattering'], color='red')
# # plt.plot(new_prediction_h100r60['lambda'], new_prediction_h100r60['prediction_label'], linestyle='--', color='blue')
# # #
# # plt.text(0.95, 0.95, f'Error at Max Value: {error:.4f}', ha='right', va='top', transform=plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5'))
# # plt.text(0.95, 0.85, f'Peak Value (Prediction): {max_pred:.4f}', ha='right', va='top', transform=plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5'))
# # plt.text(0.95, 0.75, f'Peak Value (Label): {max_real:.4f}', ha='right', va='top', transform=plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5'))
# #
# # #
# # plt.title('Random Forest Regressor Predict [h100r60]')
# # plt.xlabel('Wavelength [nm]')
# # plt.ylabel('Scattering Cross Section x10^-14 [a.u]')
# # plt.show()
# # #
# # # Criar dataframe com os valores plotados
# # df_plot = pd.DataFrame({
# #     'lambda': new_prediction_h100r60['lambda'],
# #     'scattering': new_prediction_h100r60['scattering'],
# #     'prediction_label': new_prediction_h100r60['prediction_label']
# # })
# # #
# # # Salvar dataframe em um arquivo CSV
# # df_plot.to_csv('h100r60.csv', index=False)
# # # # #
# # # # #
# # """Teste de Validaçao"""
# # # #
# # # #
# # # Realizando a predição
# # new_prediction_h150_r70 = predict_model(saved_final_rf, data=data_test_h150r70)
# #
# # # Encontrando o valor máximo da predição e dos dados reais
# # max_pred = np.max(new_prediction_h150_r70['prediction_label'])
# # max_real = np.max(new_prediction_h150_r70['scattering'])
# #
# # # Calculando o erro entre o valor máximo da predição e o valor máximo dos dados reais
# # error = np.abs((max_pred - max_real)/max_real)
# #
# # # Plotando o gráfico dos valores de predição
# # plt.plot(new_prediction_h150_r70['lambda'], new_prediction_h150_r70['scattering'], color='red')
# # plt.plot(new_prediction_h150_r70['lambda'], new_prediction_h150_r70['prediction_label'], linestyle='--', color='blue')
# #
# # plt.text(0.95, 0.95, f'Error at Max Value: {error:.4f}', ha='right', va='top', transform=plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5'))
# # plt.text(0.95, 0.85, f'Peak Value (Prediction): {max_pred:.4f}', ha='right', va='top', transform=plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5'))
# # plt.text(0.95, 0.75, f'Peak Value (Label): {max_real:.4f}', ha='right', va='top', transform=plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5'))
# #
# # plt.title('Random Forest Regressor Predict [h150r70]')
# # plt.xlabel('Wavelength [nm]')
# # plt.ylabel('Scattering Cross Section x10^-14 [a.u]')
# # plt.show()
# #
# # # Criar dataframe com os valores plotados
# # df_plot = pd.DataFrame({
# #     'lambda': new_prediction_h150_r70['lambda'],
# #     'scattering': new_prediction_h150_r70['scattering'],
# #     'prediction_label': new_prediction_h150_r70['prediction_label']
# # })
# #
# # # Salvar dataframe em um arquivo CSV
# # df_plot.to_csv('h150r70.csv', index=False)