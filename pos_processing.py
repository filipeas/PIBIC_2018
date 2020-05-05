import numpy as np
from skimage.io import imread, imsave, imshow, show
from skimage import img_as_ubyte, data
from skimage.measure import label, find_contours
# from dilation_and_erosion import *
from skimage.morphology import square, rectangle, diamond, disk, cube, octahedron, ball, octagon, star, binary_closing, binary_erosion, binary_dilation, binary_opening, convex_hull_image, square
from skimage import data, img_as_float
from skimage.util import invert
from skimage import filters
from skimage.color import rgb2gray
from skimage.measure import regionprops
from skimage.filters import gaussian, threshold_otsu
from skimage.segmentation import active_contour

import os
import logging

from imageio import imread
import matplotlib
from matplotlib import pyplot as plt

import morphsnakes as ms


def visual_callback_2d(background, fig=None):
    """
    Returns a callback than can be passed as the argument `iter_callback`
    of `morphological_geodesic_active_contour` and
    `morphological_chan_vese` for visualizing the evolution
    of the levelsets. Only works for 2D images.
    
    Parameters
    ----------
    background : (M, N) array
        Image to be plotted as the background of the visual evolution.
    fig : matplotlib.figure.Figure
        Figure where results will be drawn. If not given, a new figure
        will be created.
    
    Returns
    -------
    callback : Python function
        A function that receives a levelset and updates the current plot
        accordingly. This can be passed as the `iter_callback` argument of
        `morphological_geodesic_active_contour` and
        `morphological_chan_vese`.
    
    """
    
    # Prepare the visual environment.
    if fig is None:
        fig = plt.figure()
    fig.clf()
    ax1 = fig.add_subplot(1, 2, 1)
    # ax1.imshow(background, cmap=plt.cm.gray)

    ax2 = fig.add_subplot(1, 2, 2)
    ax_u = ax2.imshow(np.zeros_like(background), vmin=0, vmax=1)
    # plt.pause(0.001)

    def callback(levelset):
        
        if ax1.collections:
            del ax1.collections[0]
        ax1.contour(levelset, [0.5], colors='r')
        ax_u.set_data(levelset)
        # fig.canvas.draw()
        # plt.pause(0.001)

    return callback


def pos_processing(result_image):
	# fechamento
	selem = star(10)
	result_image = binary_closing(result_image, selem)

	# selecionando o maior elemento
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

	result_image = biggest_component_image

	imgcolor = result_image
	img = rgb2gray(result_image)

	# g(I)
	gimg = ms.inverse_gaussian_gradient(img, alpha=1000, sigma=2)

	# pegar o centro da imagem
	label_img = label(imgcolor, connectivity=imgcolor.ndim)
	props = regionprops(label_img)
	# print(props[0].equivalent_diameter)
	# exit()

	# Initialization of the level-set.
	init_ls = ms.circle_level_set(img.shape, (props[0].centroid[0], props[0].centroid[1]), props[0].equivalent_diameter)

	# Callback for visual plotting
	callback = visual_callback_2d(imgcolor)

	# MorphGAC. 
	resultado = ms.morphological_geodesic_active_contour(gimg, iterations=200, 
	                                         init_level_set=init_ls,
	                                         smoothing=2, threshold=0.3,
	                                         balloon=-1, iter_callback=callback)

	# convex-hull
	# chull = convex_hull_image(result_image)

	# gerando imagem do convex-hull
	# selem = disk(10)
	# result_image = binary_erosion(chull, selem)

	# imsave('teste.png', img_as_ubyte(resultado))

	return resultado