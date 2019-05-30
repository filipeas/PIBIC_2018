#esse script é responsável pelo calcula e separação de cada superpixel da imagem
#ao termino no processo, ele salva dados em um arquivo do numpy com uma matriz para cada superpixel

from skimage.segmentation import slic, mark_boundaries
from skimage.io import imread, imsave, imshow, show
from skimage import img_as_ubyte
from utils import *
from superpixels_extraction import *
import sys, time, multiprocessing as mp
import numpy as np

#a função recebe o caminho da imagem original e da imagem marcada
def vsf(path, path_marked):

    #lendo as imagens
    img_original = img_as_ubyte(imread(path))
    img_marked = img_as_ubyte(imread(path_marked))

    #calculando os superpixels utilizando o slic da biblioteca skimage
    segments = slic(img_original, n_segments=1500)

    #salva a imagem dos superpixels na pasta de dados salvos
    imsave('saved_data/superpixels.png', mark_boundaries(img_original, segments))

    #segmenta as áreas doentes e saudáveis com base na marcação do médico
    #função disponível no arquivo utils.py
    healthy, disease = image_color_segmentation(img_marked)

    #recebe a imagem e as áreas doentes e saudáveis e retorna quais superpixels estão em áreas doentes e quais estão em áreas saudáveis
    #função disponível no arquivo utils.py
    healthy_superpixels, disease_superpixels = select_superpixels(segments, healthy, disease)

    #cria duas filas para executar o procedimento em paralelo
    """ pool_ht = mp.Pool(2)
    pool_ds = mp.Pool(2) """

    #gera as entradas para as filas de execução
    ht_inputs = [(img_original, segments, i) for i in healthy_superpixels]
    ds_inputs = [(img_original, segments, i) for i in disease_superpixels]

    #executa as entradas nas filas de execução
    #executa a função extract_superpixel(disponível no arquivo superpixels_extraction.py) para cada superpixel
    #a função recebe a numeração de um superpixel e retorna uma imagem que corresponde à area daquele superpixel
    """ healthy_boxes = pool_ht.map(extract_superpixel, ht_inputs)
    disease_boxes = pool_ds.map(extract_superpixel, ds_inputs) """

    healthy_boxes = list(map(extract_superpixel, ht_inputs))
    disease_boxes = list(map(extract_superpixel, ds_inputs))

    #salva dados dos superpixels num arquivo numpy
    #os dados são: superpixels doentes (números), superpixels saudáveis (números), todos os superpixels gerados pelo slic, superpixels saudáveis (imagem), superpixels doentes (imagem)
    superpixels_file = open('saved_data/superpixels.npz', 'wb')
    np.savez(superpixels_file, ht_superpixels=healthy_superpixels, ds_superpixels=disease_superpixels, segments=segments, ht_boxes=healthy_boxes, ds_boxes=disease_boxes)
    superpixels_file.close()

    #salva dados das áres resultantes da segmentação pela marcação
    #os dados são: área saudável e área doente
    superpixels_file = open('saved_data/marked_areas.npz', 'wb')
    np.savez(superpixels_file, healthy=healthy, disease=disease)
    superpixels_file.close()