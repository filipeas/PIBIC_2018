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
from SFc_means import * # classe responsável por implementar o algoritmo proposto, SFc-means
from utils import * # classe responsável por guardar funções gerais
from skimage.io import imread, imsave, imshow, show
from skimage import img_as_ubyte
from sklearn.metrics import accuracy_score
from scipy.ndimage.morphology import binary_fill_holes
from skimage.measure import label, find_contours

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
def select_random_seeds(percentage=0.3):
    ht, ds, ht_sp, ds_sp, segments = read_features_file()
    #ht_cls = [(1, i) for i in ht] #healthy features with classes
    #ds_cls = [(2, i) for i in ds] #disease features with classes

    # preparando para cortar o resultado do vetor retornado do SFC-means
    tamanho_ht = len(ht)
    tamanh_ds = len(ds)

    minimun = min(len(ht), len(ds))
    absolute_limiar = round(minimun * percentage)

    attr_vector = np.concatenate((ht, ds))
    k = 2
    seeds = np.concatenate((ht[:absolute_limiar], ds[:absolute_limiar]))

    ht_classes = [1 for i in range(absolute_limiar)]
    ht_classes.extend([2 for i in range(absolute_limiar)])
    seeds_classes = np.array(ht_classes)

    centroids = [i for i in range(absolute_limiar)]
    centroids.extend([i for i in range(len(ht), len(ht) + absolute_limiar)])
    centroids = np.array(centroids)
    
    threshold = 0.1
    z = 2
    total_grouping = 1 # 0 para agrupar em 3 grupos diferentes

    result = SFc_means(attr_vector, k, seeds, seeds_classes, centroids, threshold, z, total_grouping)

    # substituindo zeros da imagem(acredito que quando o sfc-means retorna o vetor, ele tras as caracteristicas de 3 grupos diferentes: saudáveis, não saudáveis e aleatório. assim o vetor result acaba obtendo valores 0, 1 e 2, onde o valor 0 atrapalha no calculo)
    # a outra possibilidade é que o vetor é inicializado com zeros, como ocorre na classe SFc_menas.py, só que alguns zeros não são substituido
    # for i in range(len(result)):
    #     if result[i] == 0:
    #         result[i] = 1

    # cortando a parte saudável
    ht_corte = result[0:tamanho_ht]
    ds_corte = result[tamanho_ht:]

    marks_file = open('saved_data/marked_areas.npz', 'rb')
    marks_data = np.load(marks_file)
    healthy_mark = marks_data['healthy']
    disease_mark = marks_data['disease']
    
    # cria imagem do treinamento resultante
    full_sp = np.concatenate((ht_sp, ds_sp))
    create_result_image(segments, list(zip(full_sp, result)), 'tcc_result')

    result_healthy_mask = mask_result_image(segments, list(zip(ht_sp, ht_corte)), 1)
    result_disease_mask = mask_result_image(segments, list(zip(ds_sp, ds_corte)), 2)
    pos_processed = image_pos_processing(result_disease_mask)
    #imsave('saved_data/result/pos_processed_mask.png', img_as_ubyte(pos_processed))

    tp, tn, fp, fn = calculate_confusion_matrix(disease_mark, pos_processed, healthy_mark, result_healthy_mask)
    dice = calculate_dice(disease_mark, pos_processed)
    accuracy = (tp + tn) / (tp + fp + tn + fp)
    sensibility = (tp) / (tp + fn)
    specificity = (tn) / (tn + fp)

    return accuracy, sensibility, specificity, dice
    #full_sp = np.concatenate((ht_sp, ds_sp))
    #create_result_image(segments, list(zip(full_sp, result)))

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
    """ 
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
    #create_result_image(segments, list(zip(full_sp, result_healthy)), 'train_image')
    tcc_train_image(segments, list(zip(full_sp, result_healthy)), 'train_image')
    """ 
    # sfcmeans = select_random_seeds(percentage) # usando SFc-means

    # print("teste com sfcmeans")
    # print(sfcmeans)

    clf = RandomForestClassifier(n_estimators=30) # usando Random Forest
    
    clf.fit(train_set_data, train_ground_truth)

    result_healthy_test = []
    for i in healthy_test_set_index:
        prediction = clf.predict(healthy_props[i].reshape(1, -1))
        result_healthy_test.append(prediction[0])
    
    result_disease_test = []
    for i in disease_test_set_index:
        prediction = clf.predict(disease_props[i].reshape(1, -1))
        result_disease_test.append(prediction[0])


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

    # testes
    # print("matriz result_healthy")
    # print(result_healthy)
    # print("matriz result_disease")
    # print(result_disease)

    marks_file = open('saved_data/marked_areas.npz', 'rb')
    marks_data = np.load(marks_file)
    healthy_mark = marks_data['healthy']
    disease_mark = marks_data['disease']
    
    # cria imagem do treinamento resultante
    result_healthy.extend(result_disease)

    full_sp = np.concatenate((healthy_sp, disease_sp))
    create_result_image(segments, list(zip(full_sp, result_healthy)), 'tcc_result')
    
    result_healthy_mask = mask_result_image(segments, list(zip(healthy_sp, result_healthy)), 1)
    result_disease_mask = mask_result_image(segments, list(zip(disease_sp, result_disease)), 2)
    pos_processed = image_pos_processing(result_disease_mask)
    #imsave('saved_data/result/pos_processed_mask.png', img_as_ubyte(pos_processed))

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
<b></b>:
@ params:
@ returns: 
"""
def calculate_dice(true_disease, predict_disease):
    dice = np.sum(predict_disease[true_disease]*2.0 /(np.sum(predict_disease) + np.sum(true_disease)))
    return dice

"""
<b></b>:
@ params:
@ returns: 
"""
def image_pos_processing(result_image):
    shape = result_image.shape
    labels = label(result_image) #get the image componentes labels 
    unique, counts = np.unique(labels, return_counts=True) #count the number of pixels in each label
    labeled_qtd = dict(zip(unique, counts)) #zip the labels and the quantities and put all this in a dict
    del labeled_qtd[0] #remove the key corresponding to the background

    sorted_labels = sorted(labeled_qtd, key=labeled_qtd.get, reverse=True)  #sort the labels in decrescent order to get the biggest label

    #mount, based on the biggest label, an image with only the biggest component
    biggest_component_image = np.zeros((shape[0], shape[1]), dtype=int)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if labels[i][j] == sorted_labels[0]:
                biggest_component_image[i][j] = 255

    #imsave('saved_data/result/tcc_biggest_component.png', img_as_ubyte(biggest_component_image))

    #get the contours of all image elements
    contours = find_contours(biggest_component_image, 0.5)
    #order the contours in decrescent order and get the biggest one
    biggest_contour = sorted(contours, key=lambda x: len(x), reverse=True)[0]

    #mount an image with the biggest contour
    output = np.zeros((shape[0], shape[1]), dtype=int)
    for i in biggest_contour:
        output[int(i[0])][int(i[1])] = 255

    #imsave('saved_data/result/tcc_contour.png', img_as_ubyte(output))

    #fill the biggest contour
    output = binary_fill_holes(output)

    #imsave('saved_data/result/tcc_contour_filed.png', img_as_ubyte(output))

    return output

#if __name__ == '__main__':
   # classify() # o original do trabalho do pablo era classify().
    # select_random_seeds() # Esse trabalho realiza a classificacao usando o sfc-means