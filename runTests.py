#Esse script controla a execução de todas as partes do algoritmo
#É nele que são lidas as imagens e estas são passadas para as demais etapadas do algoritmo
#Ao término no processo, ele gera arquivos .csv (tabelas) com os resultados individuais de cada imagem e a média geral de todas as execuções

from main import vsf
from features_extraction import ftet
from select_and_classify import classify
from statistics import mean
import os, csv, time
import numpy as np

#imagens que irão passar pela execução do algoritmo
dataset = [
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    '10',
    '11',
    '13',
    '14',
    '15',
    '16',
    '17',
    '18',
    '19',
    '20'
]

#porcentagens de dados das imagens usadas para treino
percentages = [
    0.01,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.5
]

#tenta encontrar o diretório results. Se ele não existir, o cria
results_path = 'results/'
if not os.path.exists(results_path):
    os.makedirs(results_path)

#cria a estrutura de dados que irá guardar os resultados das médias das execuções para todas as imagens e para todas as métricas
#funciona como uma matriz  de (Nª de porcentagenx utilizadas)x(número de métricas, que no caso são 4)
#no final, essa estrutura é utilizada para calcular a média das médias de cada porcentagem de dados pra treino
total_means = [[[],[],[],[]] for i in range(len(percentages))]

#abre um arquivo .csv para escrever os resultados gerais
#escreve apenas o cabeçalho por enquanto
general = open(results_path + 'general.csv', 'w')
total_results = csv.writer(general, delimiter=',')
total_results.writerow(['', 'Accuracy', '','Sensitivity', '', 'Specificity', ''])
total_results.writerow(['', 'Mean', 'Std','Mean', 'Std', 'Mean', 'Std'])

#executando para cada imagem
for image in dataset:
    print(f'\n\n###########Doing for image {image}############\n\n')

    #marcando o tempo de execução
    start = time.time()

    #abrindo a imagem original e a imagem marcada
    path = f'Database/{image}/{image}_orig'
    path_marked = f'Database/{image}/{image}'

    #executa a primeira etapa do algoritmo, responsável pela separação dos superpixels da imagem (arquivo main.py)
    vsf(path, path_marked)

    #executa a segunda etapa do algoritmo, responsável pela extração de características de cada superpixel gerado na etapa anterior
    #disponível no arquivo features_extraction.py
    ftet()

    #abre um arquivo .csv para escrever os resultados individuais da imagens
    #escreve apenas o cabeçalho por enquanto
    results_image = open(f'results/results_{image}.csv', 'w')
    image_results = csv.writer(results_image, delimiter=',')
    image_results.writerow(['Percentage', 'Accuracy', 'Sensibility', 'Specificity', 'Dice'])

    #para cada percentual
    for index, percent in enumerate(percentages):
        
        #cria uma estrutura de dados para salvar os resultados e posteriormente calcular as médias das métricas
        metrics_media = [[], [], [], []]

        #executa 5 vezes para depois tirar a média
        for i in range(5):

            #terceira etapa do algoritmo
            #usa as características extraídas na etapa anterior para classificar e gerar as métricas para cada imagem
            #disponível no arquivo select_and_classufy.py
            acc, sen, spe, dice = classify(percent)

            #coloca as métricas na estrutura de dados
            metrics_media[0].append(acc)
            metrics_media[1].append(sen)
            metrics_media[2].append(spe)
            metrics_media[3].append(dice)

        #ao final das 5 execuções, escreve no arquivo .csv a média dos resultados
        image_results.writerow([f'{percent}', f'{mean(metrics_media[0]*100)}', f'{mean(metrics_media[1]*100)}', f'{mean(metrics_media[2]*100)}', f'{mean(metrics_media[3])}'])

        #depois adiciona à estrutura de dados com os dados gerais        
        total_means[index][0].append(mean(metrics_media[0])*100)
        total_means[index][1].append(mean(metrics_media[1])*100)
        total_means[index][2].append(mean(metrics_media[2])*100)
        total_means[index][3].append(mean(metrics_media[3]))
    
    #fecha a imagem e calcula o tempo de execução do algoritmo para a imagem
    results_image.close()
    end = time.time()
    print(f'\n\n###########Done for image {image} in {end-start} seconds############\n\n')

#ao final da execução de todas as imagens, escreve no arquivo .csv de médias gerais, calculando-as
for index, percent in enumerate(total_means):
    total_results.writerow([str(percentages[index]), str(mean(percent[0])), str(np.std(percent[0])),str(mean(percent[1])), str(np.std(percent[1])), str(mean(percent[2])), str(np.std(percent[2])), str(mean(percent[3])), str(np.std(percent[3]))])

#fecha o arquivo de médias gerais
general.close()