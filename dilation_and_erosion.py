# Este arquivo tem a função de implementar alguns métodos de erosão
# e dilatação da opencv para utilizar no pós processamento da máscara
# resultante da lesão.
#
# author:
# Filipe A. Sampaio - https://github.com/filipeas
#
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def pegaImagem(imagem):
	return cv2.imread(imagem, 0) # 0 para converter em escala de cinza


# dilatacao do opencv (tem efeito de erosão)
def erosao(img):
	kernel = np.ones((20,20), np.uint8)
	dilatacao = cv2.dilate(img, kernel, iterations = 1)
	return dilatacao


# erosao do opencv (tem efeito de dilatacao)
def dilatacao(img):
	kernel = np.ones((20,20), np.uint8)
	eros = cv2.erode(img, kernel, iterations = 1)
	return eros


# fechamento seguido de abertura (dilatacao seguido de erosao. depois erosao seguido de dilatacao)
def fechamentoComAbertura(img):
	kernel = np.ones((22,22), np.uint8)
	closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
	opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
	return opening


# mostra resultado na tela
def printaResultado(img, imgResultante):
	cv2.imshow('original', img)
	cv2.imshow('resultado', imgResultante)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

######################### TESTES ##############################

# imagens = [
# 	'pos.png',
# 	'ant.png',
# 	'pos2.png',
# 	'ant2.png',
# ]

# img = pegaImagem(imagens[3])

# # imgResultante = erosao(dilatacao(img))
# imgResultante = fechamentoComAbertura(img)

# printaResultado(img, imgResultante)

'''
resumo:

o melhor metodo observado foi o fechamentoComAbertura,
pois ele manteve proporção da lesão como ela realmente é
na imagem original.
'''