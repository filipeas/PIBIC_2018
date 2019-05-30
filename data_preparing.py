# <b>data_preparing</b>: Classe responsável somente por receber um array contendo os superpixels e extrai as caracteristicas
# necessárias para a classificação.
# 
# author:
# Pablo Vinicius - https://github.com/pabloVinicius
#
# contributors:
# Filipe A. Sampaio - https://github.com/filipeas

#imports necessários
from skimage.feature import greycomatrix, greycoprops
from skimage.color import rgb2gray
from skimage import img_as_ubyte
from skimage.measure import shannon_entropy
import numpy as np

"""
<b>extract_features</b>: Extrai as caracteristicas (features) de cada superpixel.
@ params:
array -> imagem com superpixels
@ returns:
props_array -> array de caracteristicas (contrast, energy, homogeneity, correlation)
"""
def extract_features(array):
    props_array = []
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    for img in array:
        ubyte_img = img_as_ubyte(img)
        img_features = np.array([])
        for i in range(3):
            glcm = greycomatrix(ubyte_img[:,:,i], [1], angles)
            filt_glcm = glcm[:, :, :, :]
            constrast = greycoprops(filt_glcm, 'contrast')
            energy = greycoprops(filt_glcm, 'energy')
            homogeneity = greycoprops(filt_glcm, 'homogeneity')
            correlation = greycoprops(filt_glcm, 'correlation')
            entropy = shannon_entropy(ubyte_img[:,:,i])
            img_features = np.insert(img_features, 0, np.concatenate((constrast, energy, homogeneity, correlation)).flatten())
            img_features = np.insert(img_features, 0, entropy)
        props_array.append(img_features)
        
    return props_array
    