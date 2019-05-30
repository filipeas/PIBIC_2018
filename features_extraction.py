#esse script é responsável por extrair características de cada um dos superpixels gerados na primeira etapa do modelo proposto
#ao final do procedimento, ele salva as features extraídas em um arquivo numpy
from data_preparing import *
import numpy as np

def ftet():
    #lẽ o arquivo gerado na etapa anterior
    superpixels_file = open('saved_data/superpixels.npz', 'rb')
    data_file = np.load(superpixels_file)

    #lê as imagens dos superpixels extraidos
    healthy_boxes = data_file['ht_boxes']
    disease_boxes = data_file['ds_boxes']

    #extrai as caracteristicas (features) de cada superpixel
    #função disponível no arquivo data_preparing.py
    healthy_props = extract_features(healthy_boxes)
    disease_props = extract_features(disease_boxes)

    #salva os dados em um arquivo numpy
    #os dados são: características dos superpixels saudáveis e características dos superpixels doentes
    features_file = open('saved_data/features.npz', 'wb')
    np.savez(features_file, healthy=healthy_props, disease=disease_props)
    features_file.close()