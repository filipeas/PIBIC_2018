# Classe principal do código
# 
# exec -> python runTests.py
# 
# <b>runTests</b>: Esse script controla a execução de todas as partes do algoritmo.
# É nele que são lidas as imagens e estas são passadas para as demais etapadas do algoritmo
# Ao término no processo, ele gera arquivos .csv (tabelas) com os resultados individuais de cada imagem
# e a média geral de todas as execuções.
# 
# author:
# Pablo Vinicius - https://github.com/pabloVinicius
#
# contributors:
# Filipe A. Sampaio - https://github.com/filipeas

# imports dos arquivos necessários
from main import vsf # responsável pela separação dos superpixels
from features_extraction import ftet # responsável por extrair as caracteristicas de cada superpixel usadas no modelo
from select_and_classify import classify # responsável pela classificação do modelo criado
from statistics import mean, pvariance
import os, csv, time
import numpy as np



# imagens que irão passar pela execução do algoritmo
dataset = [
    '1'
    # '1',
    # '2',
    # '3',
    # '4',
    # '5',
    # '6',
    # '7',
    # '8',
    # '9',
    # '10',
    # '11',
    # '13',
    # '14',
    # '15',
    # '16',
    # '17',
    # '18',
    # '19',
    # '20'
]

# Porcentagens de dados das imagens usadas para treino.
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

# Quantidade de segmentos para o superpixel
qtdSegments = [
    1500,
    2000,
    2500
]

# Tenta encontrar o diretório results. Se ele não existir, o cria.
results_path = 'results/'
if not os.path.exists(results_path):
    os.makedirs(results_path)

# Cria a estrutura de dados que irá guardar os resultados das médias das execuções para todas as imagens e para todas as métricas.
# Funciona como uma matriz  de (Nª de porcentagenx utilizadas)x(número de métricas, que no caso são 4).
# No final, essa estrutura é utilizada para calcular a média das médias de cada porcentagem de dados pra treino.
total_means = [[[],[],[],[]] for i in range(len(percentages))]

# Abre um arquivo .csv para escrever os resultados gerais.
# Escreve apenas o cabeçalho por enquanto.
general = open(results_path + 'general.csv', 'w')
total_results = csv.writer(general, delimiter=',')
total_results.writerow(['', 'Accuracy', '','Sensitivity', '', 'Specificity', '', 'Dice', ''])
total_results.writerow(['', 'Mean', 'Standard Deviation', 'Std', '', 'Mean', 'Standard Deviation', 'Std', '', 'Mean', 'Standard Deviation', 'Std', '', 'Mean', 'Standard Deviation', 'Std'])

# Executando para cada imagem
for image in dataset:
    print(f'\n\n########### Executando a imagem {image} ############\n\n')

    # Marcando o tempo de execução.
    start = time.time()

    # Abrindo a imagem original e a imagem marcada.
    path = f'Database/{image}/{image}_orig' # original
    path_marked = f'Database/{image}/{image}' # marcada

    # Abre um arquivo .csv para escrever os resultados individuais 
    # da imagens(porcentagem usada, acurácia, sensibilidade, especificidade e Dice).
    # Escreve apenas o cabeçalho, por enquanto.
    results_image = open(f'results/results_{image}.csv', 'w')
    image_results = csv.writer(results_image, delimiter=',')
    image_results.writerow(['Percentage', 'Segment', 'Accuracy', 'Sensibility', 'Specificity', 'Dice', 'Standard Deviation'])

    # iterando sobre a quantidade de segmentos de imagem para o superpixel
    for segment in qtdSegments:
        # Executa a primeira etapa do algoritmo, responsável pela separação dos superpixels da imagem (arquivo main.py).
        vsf(path, path_marked, segment)

        # Executa a segunda etapa do algoritmo, responsável pela extração de características de cada superpixel gerado na etapa anterior.
        # Disponível no arquivo features_extraction.py
        ftet()

        # Para cada percentual:
        for index, percent in enumerate(percentages):
            
            # Cria uma estrutura de dados para salvar os resultados e posteriormente calcular as médias das métricas
            metrics_media = [[], [], [], []]

            # Executa 5 vezes para depois tirar a média (5 vezes pois são 5 métricas diferentes)
            for i in range(5):

                # Terceira etapa do algoritmo:
                # Usa as características extraídas na etapa anterior para classificar e gerar as métricas para cada imagem
                # Disponível no arquivo select_and_classify.py (alteração por Filipe A.S: classificar usando SFc-means).
                acc, sen, spe, dice = classify(percent)

                # Coloca as métricas na estrutura de dados
                metrics_media[0].append(acc)
                metrics_media[1].append(sen)
                metrics_media[2].append(spe)
                metrics_media[3].append(dice)

            desvio_padrao = []
            for i in range(len(metrics_media)):
                for j in range(len(metrics_media[i])):
                    desvio_padrao.append(metrics_media[i][j])

            # Ao final das 5 execuções, escreve no arquivo .csv a média dos resultados
            image_results.writerow([f'{percent}', f'{segment}', f'{mean(metrics_media[0]*100)}', f'{mean(metrics_media[1]*100)}', f'{mean(metrics_media[2]*100)}', f'{mean(metrics_media[3])}', f'{pvariance(desvio_padrao)}'])

            # Depois adiciona na estrutura de dados os dados gerais        
            total_means[index][0].append(mean(metrics_media[0])*100) # acuracia
            total_means[index][1].append(mean(metrics_media[1])*100) # sensibilidade
            total_means[index][2].append(mean(metrics_media[2])*100) # especificidade
            total_means[index][3].append(mean(metrics_media[3])) # dice
    
    # Fecha a imagem e calcula o tempo de execução do algoritmo para a imagem atual.
    results_image.close()
    end = time.time()
    print(f'\n\n########### Fim da execucao da imagem {image} no tempo de {end-start} segundos ############\n\n')

# Ao final da execução de todas as imagens, escreve no arquivo .csv de médias gerais, calculando-as
for index, percent in enumerate(total_means):
    total_results.writerow([str(percentages[index]), str(mean(percent[0])), str(pvariance(percent[0])), str(np.std(percent[0])),str(mean(percent[1])), str(pvariance(percent[1])), str(np.std(percent[1])), str(mean(percent[2])), str(pvariance(percent[2])), str(np.std(percent[2])), str(mean(percent[3])), str(pvariance(percent[3])), str(np.std(percent[3]))])

# Fecha o arquivo de médias gerais
general.close()