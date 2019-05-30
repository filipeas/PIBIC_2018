# <b>features_extraction</b>: Esse script é responsável por extrair características de cada um dos superpixels
# gerados na primeira etapa do modelo proposto.
# So final do procedimento, ele salva as features extraídas em um arquivo numpy.
# 
# author:
# Pablo Vinicius - https://github.com/pabloVinicius
#
# contributors:
# Filipe A. Sampaio - https://github.com/filipeas

# imports necessários
from data_preparing import *
import numpy as np

def ftet():

    # Lê o arquivo gerado na etapa anterior:
    superpixels_file = open('saved_data/superpixels.npz', 'rb')
    data_file = np.load(superpixels_file)

    # Lê as imagens dos superpixels extraidos
    healthy_boxes = data_file['ht_boxes']
    disease_boxes = data_file['ds_boxes']

    # Extrai as caracteristicas (features) de cada superpixel
    # Função disponível no arquivo data_preparing.py (função que de fato extrai as caracteristicas)
    healthy_props = extract_features(healthy_boxes)
    disease_props = extract_features(disease_boxes)

    # Salva os dados em um arquivo numpy
    # Os dados são: características dos superpixels saudáveis e características dos superpixels doentes
    features_file = open('saved_data/features.npz', 'wb')
    np.savez(features_file, healthy=healthy_props, disease=disease_props)
    features_file.close()