#<b>utils</b>: Funções utilizadas por outros scripts no decorrer da execução do modelo.
# 
# author:
# Pablo Vinicius - https://github.com/pabloVinicius
#
# contributors:
# Filipe A. Sampaio - https://github.com/filipeas

# imports necessários
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes
from skimage.io import imshow, show, imsave, imread
from skimage import img_as_ubyte


"""
<b>image_color_segmentation</b>: Função responsável por segmentar as áreas doentes e saudáveis com base na marcação do médico

@ params:
img -> imagem para segmentação
@ returns:
healthy_mask -> máscara da imagem da parte saudável
disease_mask -> máscara da imagem da parte lesionada
"""
def image_color_segmentation(img):
    shape = img.shape
    
    healthy_mask = np.zeros((shape[0],shape[1]),  dtype=int)
    disease_mask = np.zeros((shape[0],shape[1]),  dtype=int)

    for i in range(shape[0]):
        for j in range(shape[1]):
            if(img[i][j][0] >= 0 and img[i][j][0] <= 50 and img[i][j][1] >= 0 and img[i][j][1] <= 50  and img[i][j][2] >= 200 and img[i][j][2] <= 255):
                healthy_mask[i][j] = 1
            elif(img[i][j][0] >= 230 and img[i][j][0] <= 255 and img[i][j][1] >= 0 and img[i][j][1] <= 84 and img[i][j][2] >= 0 and img[i][j][2] <= 68):
                disease_mask[i][j] = 1

    healthy_mask = binary_fill_holes(healthy_mask)
    disease_mask = binary_fill_holes(disease_mask)
    
    healthy_mask = healthy_mask ^ disease_mask

    return healthy_mask, disease_mask

"""
<b>select_superpixels</b>: Função responsável por receber a imagem e as áreas (doentes e saudáveis),
e retorna quais superpixels estão em áreas doentes e quais estão em áreas saudáveis.
@ params:
segments -> imagem com os superpixels
healthy_mask -> máscara com a parte saudável
disease_mask -> máscara com a parte lesionada
@ returns:
final_ht_segments -> segmento indicando quais superpixels estão saudáveis
final_ds_segments -> segmento indicando quais superpixels estão lesionados
"""
def select_superpixels(segments, healthy_mask, disease_mask):
    shape = segments.shape

    healthy_segments = []
    disease_segments = []

    for i in range(shape[0]):
        for j in range(shape[1]):
            if healthy_mask[i][j]:
                healthy_segments.append(segments[i][j])
            if disease_mask[i][j]:
                disease_segments.append(segments[i][j])
    ht_unique, ht_counts = np.unique(healthy_segments, return_counts=True)
    ds_unique, ds_counts = np.unique(disease_segments, return_counts=True)

    final_ht_segments = []
    final_ds_segments = []

    for index,value in enumerate(ht_unique):
        if value in ds_unique:
            if ht_counts[index] > ds_counts[np.where(ds_unique == value)]:
                final_ht_segments.append(value)
            else:
                final_ds_segments.append(value)
        else:
            final_ht_segments.append(value)
    
    for value in ds_unique:
        if value not in ht_unique:
            final_ds_segments.append(value)
    
    return final_ht_segments, final_ds_segments

"""
Função responsável por criar uma imagem resultante da classificação. Essa imagem é gerada antes do pós processamento.
Parametros:
segmentos da imagem;
lista zipada com as classes, que são a lesão e a não lesão;
nome do arquivo que será gerado. por padrão se chamará tst
"""
def create_result_image(segments, classes, name='tst'):
    shape = segments.shape
    result = np.zeros((shape[0], shape[1], 3), dtype=int)
    result[:,:,2].fill(255)
    ones = [i[0] for i in classes if i[1] == 1]
    twos = [i[0] for i in classes if i[1] == 2]

    for i in range(shape[0]):
        for j in range(shape[1]):
            if segments[i][j] in ones: # para a classe 1 será colocado pixels com cor verde
                result[i][j][2] = 0
                result[i][j][1] = 255
            elif segments[i][j] in twos: # para a classe 2 será colocado pixels com cor vermelho
                result[i][j][2] = 0
                result[i][j][0] = 255

    imsave('saved_data/result/' + name + '.png', img_as_ubyte(result))

"""
Função responsável por criar uma imagem resultante da classificação, onde essa imagem será terá apenas o fundo verde da classe saudável.
Ela será usada na função override_ask, que irá adicionar a imagem da lesão.
Parametros:
segmentos da imagem;
lista zipada com as classes, que são a lesão e a não lesão;
nome do arquivo que será gerado. por padrão se chamará tst2
"""
def create_result_image_pos_processing(segments, classes, name='tst2'):
    shape = segments.shape
    result = np.zeros((shape[0], shape[1], 3), dtype=int)
    result[:,:,2].fill(255)
    ones = [i[0] for i in classes if i[1] == 1]
    twos = [i[0] for i in classes if i[1] == 2]

    for i in range(shape[0]):
        for j in range(shape[1]):
            if segments[i][j] in ones:# para a classe 1 será colocado pixels com cor verde
                result[i][j][2] = 0
                result[i][j][1] = 255
            elif segments[i][j] in twos:# para a classe 1 será colocado pixels com cor verde
                result[i][j][2] = 0
                result[i][j][1] = 255

    imsave('saved_data/result/' + name + '.png', img_as_ubyte(result))

"""
Função responsável por gerar uma imagem do treino da imagem.
Parametros:
segmentos da imagem;
lista zipada com as classes, que são a lesão e a não lesão;
nome do arquivo que será gerado. por padrão se chamará tst2
"""
def tcc_train_image(segments, classes, name='tst'):
    shape = segments.shape
    result = img_as_ubyte(imread('saved_data/superpixels.png'))
    
    ones = [i[0] for i in classes if i[1] == 1]
    twos = [i[0] for i in classes if i[1] == 2]

    for i in range(shape[0]):
        for j in range(shape[1]):
            if segments[i][j] in ones:
                result[i][j][2] = 0
                result[i][j][1] = 255
                result[i][j][0] = 0
            elif segments[i][j] in twos:
                result[i][j][2] = 0
                result[i][j][1] = 0
                result[i][j][0] = 255

    imsave('saved_data/result/' + name + '.png', img_as_ubyte(result))

def override_result(original, segments, classes, name='ovr'):
    shape = segments.shape
    result = original
    ones = [i[0] for i in classes if i[1] == 1]
    twos = [i[0] for i in classes if i[1] == 2]

    for i in range(shape[0]):
        for j in range(shape[1]):
            if segments[i][j] in twos:
                result[i][j][2] = 0
                result[i][j][1] = 0
                result[i][j][0] = 255

    imsave('saved_data/' + name + '.png', img_as_ubyte(result))

def mask_result_image(segments, classes, true_value):
    shape = segments.shape
    result = np.zeros((shape[0], shape[1]), dtype=int)

    selecteds = [i[0] for i in classes if i[1] == true_value]
    for i in range(shape[0]):
        for j in range(shape[1]):
            if segments[i][j] in selecteds:
                result[i][j] = 255

    return result

"""
Função responsável por adicionar a imagem da lesão em cima da imagem do fundo do olho.
Parametros:
imagem original;
imagem da mascara da lesao;
"""
def override_mask(original, mask):
    shape = original.shape
    result = original

    for i in range(shape[0]):
        for j in range(shape[1]):
            if mask[i][j]: # onde for pixels que representa a lesão, na imagem final será representada como cor vermelha;
                result[i][j][0] = 255
                result[i][j][1] = 0
                result[i][j][2] = 0

    return result

def select_random_seeds(array, percentage=0.3):
    pass