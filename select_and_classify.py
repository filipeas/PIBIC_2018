# <b>select_and_classify</b>: Esse script contém as funções de classificação do modelo.
# Em geral, é utilizada a função classify(), que classifica e realiza o pós-processamento
# para depois calcular as métricas. Ao final do processo, ela retorna 4 métricas:
# acurácia, sensibilidade, especificidade e medida Dice.

# 
# author:
# Pablo Vinicius - https://github.com/pabloVinicius
#
# contributors:
# Filipe A. Sampaio - https://github.com/filipeas
import numpy as np
from random import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from SFc_means import * # classe responsável por implementar o algoritmo proposto, SFc-means
from utils import * # classe responsável por guardar funções gerais
from scipy.ndimage.morphology import binary_fill_holes
from skimage.io import imread, imsave, imshow, show
from skimage import img_as_ubyte, data
from skimage.measure import label, find_contours
# from dilation_and_erosion import *
from skimage.morphology import disk, binary_closing, binary_erosion, binary_dilation, binary_opening, convex_hull_image, square
from skimage import data, img_as_float
from skimage.util import invert
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_otsu
from skimage.segmentation import active_contour
from pos_processing import pos_processing # classe responsável por realizar pós processamento da imagem lesionada da classificação

"""
<b>read_features_file</b>: Função responsável por realizar leitura do arquivo numpy que contém todas as caracteristicas
que serão utilizadas na classificação.
@ params:

@ returns: 
retorna todas as caracteristicas
"""
def read_features_file():
    sp_file = open('saved_data/superpixels.npz', 'rb')
    sp_data = np.load(sp_file)
    ft_file= open('saved_data/features.npz', 'rb')
    np_file = np.load(ft_file)
    #ft_file.close()
    return np_file['healthy'], np_file['disease'], sp_data['ht_superpixels'], sp_data['ds_superpixels'], sp_data['segments']

"""
<b></b>:
@ params:
@ returns: 
"""
def split_classes(array, group):
    return [(group, i) for i in array]

"""
<b>select_random_seeds</b>: Função responsável por chamar o classificador sfc-means
@ params:
@ returns: 
"""
def select_random_seeds(numberImage, percentage=0.3):
    ht, ds, ht_sp, ds_sp, segments = read_features_file()
    # ht_cls = [(1, i) for i in ht] #healthy features with classes
    # ds_cls = [(2, i) for i in ds] #disease features with classes

    # preparando para cortar o resultado do vetor retornado do SFC-means
    tamanho_ht = len(ht)
    tamanh_ds = len(ds)

    minimun = min(len(ht), len(ds))
    # correção: colocado o valor arredondado ou 1 para caso haja um zero
    absolute_limiar = round(minimun * percentage) or 1

    #
    # testando bloco do clasify()...
    #
    healthy_indexes = [i for i in range(0, len(ht))]
    disease_indexes = [i for i in range(0, len(ds))]

    for i in range(3):
        shuffle(healthy_indexes)
        shuffle(disease_indexes)
    
    healthy_train_set_index = healthy_indexes[:absolute_limiar]
    healthy_test_set_index = healthy_indexes[absolute_limiar:]
    disease_train_set_index = disease_indexes[:absolute_limiar]
    disease_test_set_index = disease_indexes[absolute_limiar:]

    train_set_data = []
    train_ground_truth = []

    for i in range(absolute_limiar):
        train_set_data.append(ht[healthy_train_set_index[i]])
        train_ground_truth.append(1)
        train_set_data.append(ds[disease_train_set_index[i]])
        train_ground_truth.append(2)

    train_set_data = np.asarray(train_set_data) # (sementes)
    train_ground_truth = np.asarray(train_ground_truth) # (classes das sementes)

    #gera imagem que representa como foi o dataset de treino 
    #isso foi usado apenas para gerar imagens para os artigos
    #generating train image
    result_healthy = [0 for i in range(len(ht_sp))]
    result_disease = [0 for i in range(len(ds_sp))]
    for i in healthy_train_set_index:
        result_healthy[i] = 1
    for i in disease_train_set_index:
        result_disease[i] = 2
    
    result_healthy.extend(result_disease) # (pegando as regiões de cada área e juntando em um unico vetor)

    full_sp = np.concatenate((ht_sp, ds_sp)) # (vetor de atributo de cada elemento)
    # create_result_image(segments, list(zip(full_sp, result_healthy)), 'train_image 1 (pontos usados para treino)')
#    tcc_train_image(segments, list(zip(full_sp, result_healthy)), 'train_image 2 (apontando o local de alguns pontos usados para treino)')
    # print("train_set_data", train_set_data)
    # print("train_ground_truth", train_ground_truth)
    # print("healthy_train_set_index", healthy_train_set_index)
    # print("disease_train_set_index", disease_train_set_index)
    # print("result_healthy", result_healthy)
    # print("result_disease", result_disease)
    # print("healthy_test_set_index", healthy_test_set_index)
    # print("len = ", len(healthy_test_set_index))
    # print("disease_test_set_index", disease_test_set_index)
    # print("len = ", len(disease_test_set_index))
    # exit()



    # vetor de atributos (caracteristicas)
    attr_vector = np.concatenate((ht, ds))
    
    seeds = np.concatenate((ht[:absolute_limiar], ds[:absolute_limiar]))
    seedsindex = np.concatenate((healthy_train_set_index, disease_train_set_index)) # (centroides)
    # print("seeds", seeds)
    # print("seedsindex", seedsindex)

    ht_classes = [1 for i in range(absolute_limiar)]
    ht_classes.extend([2 for i in range(absolute_limiar)])
    seeds_classes = np.array(ht_classes) # (classes de sementes)
    # print("seeds_classes", seeds_classes)

    centroids = [i for i in range(absolute_limiar)]
    centroids.extend([i for i in range(len(ht), len(ht) + absolute_limiar)])
    centroids = np.array(centroids)
    # print("centroids", centroids)
    
    k = 2 # (quantidade de grupos)
    threshold = 0.1 # 0.1 (limiar)
    z = 1.5 # (nível de nebulosidade. OBS: eu não sei oq é isso!) originalmente era 2
    total_grouping = 0 # (indica quantos grupos serão formados) 0 para agrupar em 3 grupos diferentes se baseando pelo limiar e 1 para agrupar de forma total, ou seja, usando só 2 grupos

    result = SFc_means(attr_vector, k, seeds, seeds_classes, centroids, threshold, z, total_grouping) # (método antigo - trabalho pibic 2018)
    # result = SFc_means(attr_vector, k, train_set_data, train_ground_truth, seedsindex, threshold, z, total_grouping)
    for i in range(len(result)):
        if result[i] == 0:
            result[i] = 1
    # print(result)
    # exit()

    marks_file = open('saved_data/marked_areas.npz', 'rb')
    marks_data = np.load(marks_file)
    healthy_mark = marks_data['healthy']
    disease_mark = marks_data['disease']
    
    # cria imagem do treinamento resultante
    # result_healthy.extend(result_disease)

    # print(result_healthy)

    full_sp = np.concatenate((ht_sp, ds_sp))
    create_result_image(segments, list(zip(full_sp, result)), 'tcc_result (imagem resultante da classificação)' + numberImage)

    # cortando o result de acordo com as areas que as representam
    # ht_corte = result[0:tamanho_ht]
    # ds_corte = result[tamanho_ht:]

    # create_result_image(segments, list(zip(full_sp, ht_corte)), 'ht_corte')
    # result_healthy_mask_ht_corte = mask_result_image(segments, list(zip(ht_sp, ht_corte)), 1)
    # imsave('saved_data/result/result_healthy_mask_ht_corte.png', img_as_ubyte(result_healthy_mask_ht_corte))

    # create_result_image(segments, list(zip(full_sp, ds_corte)), 'ds_corte')
    # result_healthy_mask_ht_corte = mask_result_image(segments, list(zip(ht_sp, ds_corte)), 1)
    # imsave('saved_data/result/result_healthy_mask_ds_corte.png', img_as_ubyte(result_healthy_mask_ht_corte))
    
    result_healthy_mask = mask_result_image(segments, list(zip(ht_sp, result)), 1)
    result_disease_mask = mask_result_image(segments, list(zip(ds_sp, result_disease)), 2)
    pos_processed = pos_processing(result_disease_mask)
    imsave('saved_data/result/pos_processed_mask.png', img_as_ubyte(pos_processed))
    
    # gerando imagem com o a lesão pós processada
    create_result_image_pos_processing(segments, list(zip(full_sp, result)), 'fundo_sem_lesao (usando o result)')
    # create_result_image_pos_processing(segments, list(zip(full_sp, result_healthy)), 'fundo_sem_lesao (só os pontos)')
    
    img_original_sem_lesao_completo = img_as_ubyte(imread('saved_data/result/fundo_sem_lesao (usando o result).png'))
    
    # posprocessamento na regiao verde
    # result_healthy_result_mask = image_healty_pos_processing(img_original_sem_lesao_completo.copy(), pos_processed.copy()) # teste (pode descartar depois)
    # imsave('saved_data/result/result_healthy_result_mask.png', img_as_ubyte(result_healthy_result_mask)) # teste (pode descartar depois)

    overrided2 = override_mask(img_original_sem_lesao_completo, pos_processed)
    imsave('saved_data/result/override_fundo_com_result_pos_processing.png', img_as_ubyte(overrided2))

    # img_original_sem_lesao_so_os_pontos = img_as_ubyte(imread('saved_data/result/fundo_sem_lesao (só os pontos).png'))
    # overrided1 = override_mask(img_original_sem_lesao_so_os_pontos, pos_processed)
    # imsave('saved_data/result/override_fundo_com_os_pontos_pos_processing.png', img_as_ubyte(overrided1))

    # pós-processamento sobre o result_healthy_mask:
    # selem = disk(5)
    # result_healthy_mask = binary_dilation(result_healthy_mask, selem)

    # gerando imagem resultante da mascara da area saudável
    imsave('saved_data/result/result_healthy_mask.png', img_as_ubyte(result_healthy_mask))
    # gerando imagem resultante da mascara da area lesionada
    imsave('saved_data/result/result_disease_mask.png', img_as_ubyte(result_disease_mask))

    
    tp, tn, fp, fn = calculate_confusion_matrix(disease_mark, pos_processed, healthy_mark, result_healthy_mask)
    dice = calculate_dice(disease_mark, pos_processed)
    accuracy = (tp + tn) / (tp + fp + tn + fp)
    sensibility = (tp) / (tp + fn)
    specificity = (tn) / (tn + fp)

    return accuracy, sensibility, specificity, dice

    #
    # fim do bloco do clasify()
    #

"""
<b>classify</b>: Usa as características extraídas na etapa anterior para classificar e gerar as métricas para cada imagem.
@ params:
percentage -> porcentagem de treino;
@ returns: 
retorna todas as medidas (acuracia, sensibilidade, especificidade e dice);
"""
def classify(percentage=0.25):
    healthy_props, disease_props, healthy_sp, disease_sp, segments = read_features_file()
    healthy_indexes = [i for i in range(0, len(healthy_props))]
    disease_indexes = [i for i in range(0, len(disease_props))]

    minimum = min(len(healthy_props), len(disease_props))
    absolute_limiar = round(minimum * percentage) or 1

    for i in range(3):
        shuffle(healthy_indexes)
        shuffle(disease_indexes)
    
    healthy_train_set_index = healthy_indexes[:absolute_limiar]
    healthy_test_set_index = healthy_indexes[absolute_limiar:]
    disease_train_set_index = disease_indexes[:absolute_limiar]
    disease_test_set_index = disease_indexes[absolute_limiar:]


    train_set_data = []
    train_ground_truth = []

    for i in range(absolute_limiar):
        train_set_data.append(healthy_props[healthy_train_set_index[i]])
        train_ground_truth.append(1)
        train_set_data.append(disease_props[disease_train_set_index[i]])
        train_ground_truth.append(2)

    train_set_data = np.asarray(train_set_data)
    train_ground_truth = np.asarray(train_ground_truth)
    
    #gera imagem que representa como foi o dataset de treino 
    #isso foi usado apenas para gerar imagens para os artigos
    #generating train image
    result_healthy = [0 for i in range(len(healthy_sp))]
    result_disease = [0 for i in range(len(disease_sp))]
    for i in healthy_train_set_index:
        result_healthy[i] = 1
    for i in disease_train_set_index:
        result_disease[i] = 2
    
    result_healthy.extend(result_disease)

    full_sp = np.concatenate((healthy_sp, disease_sp))
    create_result_image(segments, list(zip(full_sp, result_healthy)), 'create-image_train_image')
    tcc_train_image(segments, list(zip(full_sp, result_healthy)), 'tcc-train_train_image')
    
    # random forest
    clf = RandomForestClassifier(n_estimators=30) # usando Random Forest
    # print("clf1", clf)
    # exit()
    
    # fit do random forest
    clf.fit(train_set_data, train_ground_truth)
    # print("clf2", clf)
    # exit()

    result_healthy_test = []
    for i in healthy_test_set_index:
        prediction = clf.predict(healthy_props[i].reshape(1, -1))
        result_healthy_test.append(prediction[0])
    
    result_disease_test = []
    for i in disease_test_set_index:
        prediction = clf.predict(disease_props[i].reshape(1, -1))
        result_disease_test.append(prediction[0])

    # print("healthy_test_set_index", healthy_test_set_index)
    # print("len = ", len(healthy_test_set_index))
    # print("disease_test_set_index", disease_test_set_index)
    # print("len = ", len(disease_test_set_index))
    # exit()

    result_healthy = [0 for i in range(len(healthy_sp))]
    result_disease = [0 for i in range(len(disease_sp))]
    for index, value in enumerate(healthy_test_set_index):
        result_healthy[value] = result_healthy_test[index]
    for index, value in enumerate(disease_test_set_index):
        result_disease[value] = result_disease_test[index]
    for i in healthy_train_set_index:
        result_healthy[i] = 1
    for i in disease_train_set_index:
        result_disease[i] = 2

    # print("result_healthy",result_healthy)
    # print("result_disease",result_disease)
    # exit()

    marks_file = open('saved_data/marked_areas.npz', 'rb')
    marks_data = np.load(marks_file)
    healthy_mark = marks_data['healthy']
    disease_mark = marks_data['disease']
    
    # cria imagem do treinamento resultante
    result_healthy.extend(result_disease)
    # print("result_healthy", result_healthy)
    # exit()

    # print(result_healthy)

    full_sp = np.concatenate((healthy_sp, disease_sp))
    create_result_image(segments, list(zip(full_sp, result_healthy)), 'tcc_result')
    
    result_healthy_mask = mask_result_image(segments, list(zip(healthy_sp, result_healthy)), 1)
    result_disease_mask = mask_result_image(segments, list(zip(disease_sp, result_disease)), 2)
    pos_processed = pos_processing(result_disease_mask)
    imsave('saved_data/result/pos_processed_mask.png', img_as_ubyte(pos_processed))

    # gerando imagem do pós processamento
    create_result_image_pos_processing(segments, list(zip(full_sp, result_healthy)), 'fundo_sem_lesao')
    img_original = img_as_ubyte(imread('saved_data/result/fundo_sem_lesao.png'))
    overrided = override_mask(img_original, pos_processed)
    imsave('saved_data/result/override_pos_processing.png', img_as_ubyte(overrided))

    imsave('saved_data/result/result_healthy_mask.png', img_as_ubyte(result_healthy_mask))
    imsave('saved_data/result/result_disease_mask.png', img_as_ubyte(result_disease_mask))

    tp, tn, fp, fn = calculate_confusion_matrix(disease_mark, pos_processed, healthy_mark, result_healthy_mask)
    dice = calculate_dice(disease_mark, pos_processed)
    accuracy = (tp + tn) / (tp + fp + tn + fp)
    sensibility = (tp) / (tp + fn)
    specificity = (tn) / (tn + fp)

    return accuracy, sensibility, specificity, dice

    
    '''
    results_file = open('saved_data/result/metrics.txt', 'w')
    results_file.write(f'Acurácia: {accuracy}\n')
    results_file.write(f'Sensibilidade: {sensibility}')
    results_file.close()
    '''
    """ 
    def override_mask(original, mask):
        shape = original.shape
        result = original

        for i in range(shape[0]):
            for j in range(shape[1]):
                if mask[i][j]:
                    result[i][j][0] = 255
                    result[i][j][1] = 255
                    result[i][j][2] = 255

        return result


    img_original = img_as_ubyte(imread('test_images/2.jpg'))
    overrided = override_mask(img_original, pos_processed)
    imsave('saved_data/result/override_pos_processing.png', img_as_ubyte(overrided))
    #override_result(img_original, segments, list(zip(full_sp, result_healthy)), 'ovr_rdf_1500')
     """

    """
    healthy_grount_truth = [1 for i in range(len(healthy_sp))]
    disease_ground_truth = [2 for i in range(len(disease_sp))]
    healthy_grount_truth.extend(disease_ground_truth)

    accuracy = accuracy_score(healthy_grount_truth, result_healthy)
    print('Accuracy', accuracy)
    """
    

    """
    print('Healthy props size', len(healthy_props), len(healthy_sp))
    print('Disease props size', len(disease_props), len(disease_sp))
    print('Healthy indexes', healthy_indexes)
    print('Disease indexes', disease_indexes)
    """

"""
<b></b>:
@ params:
@ returns: 
"""
def calculate_confusion_matrix(true_disease, predict_disease, true_healthy, predict_healthy):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    shape = true_disease.shape

    for i in range(shape[0]):
        for j in range(shape[1]):
            if(true_disease[i][j]):
                if predict_disease[i][j]:
                    tp += 1
                else:
                    fn += 1
            if true_healthy[i][j]:
                if predict_healthy[i][j] == 255:
                    tn += 1
                else:
                    fp += 1
    
    return tp, tn, fp, fn

"""
<b>Função responsável por calcular o DICE</b>:
@ params:
@ returns: 
"""
def calculate_dice(true_disease, predict_disease):
    dice = np.sum(predict_disease[true_disease]*2.0 /(np.sum(predict_disease) + np.sum(true_disease)))
    return dice


"""
<b>Função responsável por realizar pós-processamento da lesão</b>: Essa função faz o pos processamento somente em cima da area lesionada.
@ params: imagem bruta da lesão
@ returns: imagem processada da lesão
"""
# def image_pos_processing(result_image):
#     # gerando imagem da lesão original
#     # imsave('saved_data/result/result_image.png', img_as_ubyte(result_image))
#     # convex-hull
#     chull = convex_hull_image(result_image)
#     # gerando imagem do convex-hull
#     # imsave('saved_data/result/chull.png', img_as_ubyte(chull))
#     selem = disk(10)
#     result_image = binary_erosion(chull, selem)

#     return result_image

"""
<b>Função responsável por realizar pós-processamento da região saudável</b>: Essa função somente faz um corte da imagem saudável com a lesão pós-processada.
@ params: img da parte saudável, img da lesão pós-processada
@ returns: img da parte saudável pós-processada
"""
# def image_healty_pos_processing(healthy_img, disease_pos_processing):
#     original = healthy_img
#     original2 = disease_pos_processing
#     shape = original.shape
#     result = original

#     for i in range(shape[0]):
#         for j in range(shape[1]):
#             if original2[i][j]: # onde for pixels que representa a lesão, na imagem final será representada como cor vermelha;
#                 result[i][j][0] = 0
#                 result[i][j][1] = 0
#                 result[i][j][2] = 0
#             if np.array_equal(original[i][j], [0,255,0]):
#                 result[i][j][0] = 255
#                 result[i][j][1] = 255
#                 result[i][j][2] = 255
#             else:
#                 result[i][j][0] = 0
#                 result[i][j][1] = 0
#                 result[i][j][2] = 0

#     return result

# def image_pos_processing(result_image):
#     # selecionando o maior elemento
#     shape = result_image.shape
#     labels = label(result_image) #get the image componentes labels 
#     unique, counts = np.unique(labels, return_counts=True) #count the number of pixels in each label
#     labeled_qtd = dict(zip(unique, counts)) #zip the labels and the quantities and put all this in a dict
#     del labeled_qtd[0] #remove the key corresponding to the background

#     sorted_labels = sorted(labeled_qtd, key=labeled_qtd.get, reverse=True)  #sort the labels in decrescent order to get the biggest label

#     #mount, based on the biggest label, an image with only the biggest component
#     biggest_component_image = np.zeros((shape[0], shape[1]), dtype=int)
#     for i in range(shape[0]):
#         for j in range(shape[1]):
#             if labels[i][j] == sorted_labels[0]:
#                 biggest_component_image[i][j] = 255

#     result_image = biggest_component_image

    
#     # realizando processo de morfologia matemática
#     selem = disk(30)
#     result_image = binary_dilation(result_image, selem)
#     selem = disk(3)
#     result_image = binary_opening(result_image, selem)
#     result_image = binary_erosion(result_image, selem)
#     selem = disk(35)
#     result_image = binary_closing(result_image, selem)
#     selem = disk(5)
#     result_image = binary_dilation(result_image, selem)
#     selem = disk(30)
#     result_image = binary_erosion(result_image, selem)

#     # imsave('saved_data/result/resultado_nova_morph.png', img_as_ubyte(result_image))

#     # realizando novamente a seleção do maior elemento
#     shape = result_image.shape
#     labels = label(result_image) #get the image componentes labels 
#     unique, counts = np.unique(labels, return_counts=True) #count the number of pixels in each label
#     labeled_qtd = dict(zip(unique, counts)) #zip the labels and the quantities and put all this in a dict
#     del labeled_qtd[0] #remove the key corresponding to the background

#     sorted_labels = sorted(labeled_qtd, key=labeled_qtd.get, reverse=True)  #sort the labels in decrescent order to get the biggest label

#     #mount, based on the biggest label, an image with only the biggest component
#     biggest_component_image = np.zeros((shape[0], shape[1]), dtype=int)
#     for i in range(shape[0]):
#         for j in range(shape[1]):
#             if labels[i][j] == sorted_labels[0]:
#                 biggest_component_image[i][j] = 255

#     # biggest_component_image = fechamentoComAbertura(img_as_ubyte(biggest_component_image))
#     # imsave('saved_data/result/tcc_biggest_component.png', img_as_ubyte(biggest_component_image))

#     # realizando o contorno ativo da area lesionada
#     #get the contours of all image elements
#     contours = find_contours(biggest_component_image, 0.5)
#     #order the contours in decrescent order and get the biggest one
#     biggest_contour = sorted(contours, key=lambda x: len(x), reverse=True)[0]

#     #mount an image with the biggest contour
#     output = np.zeros((shape[0], shape[1]), dtype=int)
#     for i in biggest_contour:
#         output[int(i[0])][int(i[1])] = 255

#     # imsave('saved_data/result/tcc_contour.png', img_as_ubyte(output))

#     # preenchendo buracos
#     #fill the biggest contour
#     output = binary_fill_holes(output)

#     # resultado final do pos processamento
#     # imsave('saved_data/result/tcc_contour_filed.png', img_as_ubyte(output))

#     return output

#if __name__ == '__main__':
   # classify() # o original do trabalho do pablo era classify().
    # select_random_seeds() # Esse trabalho realiza a classificacao usando o sfc-means