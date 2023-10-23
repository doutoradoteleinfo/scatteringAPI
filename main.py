import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle

# Função para carregar o dicionário de modelos
def load_models():
    with open('trained_model.pkl', 'rb') as file:
        model_dict = pickle.load(file)
    return model_dict

# Função para realizar a previsão
def fazer_previsao(h, r, lambdai, lambdaf, p, model):

    n_samples = int((lambdaf - lambdai) / p) + 1

    x1 = []
    for i in range(n_samples):
        x1.append([h])

    x2 = []
    for i in range(n_samples):
        x2.append([r])

    x3 = []
    for i in range(n_samples):
        valor = int(lambdai + i * p)
        x3.append([valor])

    x = np.concatenate((x1, x2, x3), axis=1)
    x = pd.DataFrame(x, columns=['altura', 'raio', 'lambda'])

    # Fazer a previsão com o modelo carregado
    prediction = model.predict(x)

    # Plotar o gráfico
    fig, ax = plt.subplots()
    ax.plot(x['lambda'], prediction, color='blue')
    ax.set_title('Random Forest Regressor Predict')
    ax.set_xlabel('Wavelength [nm]')
    ax.set_ylabel('Scattering Cross Section x10^-14[a.u]')
    # Adicionar anotações ao gráfico
    max_scattering = prediction.max()
    max_wavelength = x['lambda'][prediction.argmax()]

    ax.annotate(f"Height: {h} nm", xy=(lambdai, max_scattering), xytext=(lambdai, max_scattering), color='red', fontsize=7)
    ax.annotate(f"Radius: {r} nm", xy=(lambdai, max_scattering), xytext=(lambdai, max_scattering * 0.9), color='red', fontsize=7)
    ax.annotate(f"Wavelength peak: {max_wavelength} nm", xy=(max_wavelength, max_scattering),
                xytext=(max_wavelength + 250, max_scattering), color='red', fontsize=7)
    ax.annotate(f"Scattering peak: {max_scattering:.4f} [a.u]", xy=(max_wavelength, max_scattering),
                xytext=(max_wavelength + 250, max_scattering * 0.9), color='red', fontsize=7)

    return fig

# Configurações do Streamlit
st.set_page_config(page_title="Predict Scattering API", layout="wide")

# Dividir a tela em duas colunas
col1, col2 = st.columns(2)

# Adicionar a linha horizontal na primeira coluna
col1.markdown("<hr/>", unsafe_allow_html=True)

# Redimensionar a imagem
logo_image1 = Image.open("assets/images/logo_fotonica.jpeg")
logo_image1 = logo_image1.resize((130, 130))  # Ajuste as dimensões conforme necessário

# Criar subcolunas na primeira coluna
subcol1, subcol2 = col1.columns(2)

# Ajustar a largura da subcoluna para controlar o espaço entre a imagem e a linha horizontal
subcol1.image(logo_image1, use_column_width=150)


# Redimensionar a imagem
logo_image2 = Image.open("assets/images/logo_ufc.jpeg")
logo_image2 = logo_image2.resize((150, 150))  # Ajuste as dimensões conforme necessário

# Ajustar a largura da subcoluna para controlar o espaço entre a imagem e a linha horizontal
subcol2.image(logo_image2, use_column_width=150)

# Adicionar a linha horizontal na primeira coluna
col1.markdown("<hr/>", unsafe_allow_html=True)

# Título da página e logo
col1.title("Predict Scattering API")

# Entradas do usuário
h = col1.text_input("Enter the height of the cylindrical gold nanostructure in nm:", key="h")
if h and h.isnumeric():
    h = float(h)
r = col1.text_input("Enter the radius of the cylindrical gold nanostructure in nm:", key="r")
if r and r.isnumeric():
    r = float(r)
lambdai = col1.text_input("Enter the initial applied wavelength in nm:", key="lambdai")
if lambdai and lambdai.isnumeric():
    lambdai = float(lambdai)
lambdaf = col1.text_input("Enter the final applied wavelength in nm:", key="lambdaf")
if lambdaf and lambdaf.isnumeric():
    lambdaf = float(lambdaf)
p = col1.text_input("Enter the step of the applied wavelength in nm:", key="p")
if p and p.isnumeric():
    p = int(p)

# Botão de reset
reset_button = col1.button("Reset")

if reset_button:
    # Redefinir todas as entradas
    h = None
    r = None
    lambdai = None
    lambdaf = None
    p = None

# Carregar o dicionário de modelos
model_dict = load_models()

# Verificar se todas as caixas de entrada estão preenchidas e são números válidos
if h is not None and r is not None and lambdai is not None and lambdai != '' and lambdaf is not None and lambdaf != '' and p is not None and p != '':
    # Converta lambdai, lambdaf e p para os tipos corretos
    lambdai = float(lambdai)
    lambdaf = float(lambdaf)
    p = int(p)

    # Carregar o modelo a partir do dicionário
    model = model_dict['trained_model']  # Corrigido para 'trained_model'

    # Fazer a previsão
    fig = fazer_previsao(h, r, lambdai, lambdaf, p, model)
    # Exibir o valor do pico de scattering
    valor_pico = fig.gca().get_lines()[0].get_ydata().max()

    # Subir o gráfico e a mensagem
    col2.empty()

    # Ajustar a altura do gráfico para 300 pixels
    fig.set_size_inches(5, 4)  # Ajuste as dimensões conforme necessário

    # Exibir o gráfico
    col2.pyplot(fig)

    # Centralizar a mensagem abaixo do gráfico
    col2.text("")  # Adicionar espaço em branco
    col2.markdown(
        f'<p style="text-align: center;">The scattering peak value for this nanostructure is:</p>',
        unsafe_allow_html=True
    )
    col2.markdown(
        f'<p style="text-align: center;">{valor_pico:.4f} [a.u]</p>',
        unsafe_allow_html=True
    )



# # Entradas do usuário
# h = col1.text_input("Enter the height of the cylindrical gold nanostructure in nm:", key="h")
# if h and h.isnumeric():
#     h = float(h)
#     r = col1.text_input("Enter the radius of the cylindrical gold nanostructure in nm:", key="r")
#     if r and r.isnumeric():
#         r = float(r)
#         lambdai = col1.text_input("Enter the initial applied wavelength in nm:", key="lambdai")
#         if lambdai and lambdai.isnumeric():
#             lambdai = float(lambdai)
#             lambdaf = col1.text_input("Enter the final applied wavelength in nm:", key="lambdaf")
#             if lambdaf and lambdaf.isnumeric():
#                 lambdaf = float(lambdaf)
#                 p = col1.text_input("Enter the step of the applied wavelength in nm:", key="p")
#                 if p and p.isnumeric():
#                     p = int(p)
#                     # Verificar se todas as caixas de entrada estão vazias
#                     if h is not None and r is not None and lambdai is not None and lambdaf is not None and p is not None:
#                         # Fazer a previsão
#                         fig = fazer_previsao(h, r, lambdai, lambdaf, p)
#
#                         # Exibir o valor do pico de scattering
#                         valor_pico = fig.gca().get_lines()[0].get_ydata().max()
#
#                         # Subir o gráfico e a mensagem
#                         col2.empty()
#
#                         # Ajustar a altura do gráfico para 300 pixels
#                         fig.set_size_inches(5, 4)  # Ajuste as dimensões conforme necessário
#
#                         # Exibir o gráfico
#                         col2.pyplot(fig)
#
#                         # Centralizar a mensagem abaixo do gráfico
#                         col2.text("")  # Adicionar espaço em branco
#                         col2.markdown(
#                             f'<p style="text-align: center;">The scattering peak value for this nanostructure is:</p>',
#                             unsafe_allow_html=True
#                         )
#                         col2.markdown(
#                             f'<p style="text-align: center;">{valor_pico:.4f} [a.u]</p>',
#                             unsafe_allow_html=True
#                         )
# #
#
#
#
#
#
#
#
#
