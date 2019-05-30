from skimage.segmentation import slic, mark_boundaries
from skimage.morphology import dilation
from scipy.ndimage.morphology import binary_fill_holes
from skimage.io import imread, imsave, imshow, show
from skimage import img_as_ubyte
import sys, time, multiprocessing as mp
import numpy as np

img = img_as_ubyte(imread('test_images/2.jpg'))

shape = img.shape

healthy_mask = np.zeros((shape[0],shape[1]),  dtype=int)

for i in range(shape[0]):
    for j in range(shape[1]):
        if(img[i][j][0] >= 0 and img[i][j][0] <= 50 and img[i][j][1] >= 0 and img[i][j][1] <= 50  and img[i][j][2] >= 200 and img[i][j][2] <= 255):
            healthy_mask[i][j] = 255

healthy_mask = dilation(healthy_mask)

superpixels = img_as_ubyte(imread('saved_data/result/train_image.png'))

for i in range(shape[0]):
    for j in range(shape[1]):
        if healthy_mask[i][j] == 255:
            superpixels[i][j][0] = 0
            superpixels[i][j][1] = 0
            superpixels[i][j][2] = 255

imsave('saved_data/result/tcc_train.png', img_as_ubyte(superpixels))

img = img_as_ubyte(imread('saved_data/result/tcc_contour.png'))

for i in range(2):
    img = dilation(img)

imsave('saved_data/result/tcc_contour_dilated.png', img_as_ubyte(img))


img = img_as_ubyte(imread('saved_data/result/tcc_result.png'))
shape = img.shape

healthy_mask = np.zeros((shape[0],shape[1]),  dtype=int)

for i in range(shape[0]):
    for j in range(shape[1]):
        if(img[i][j][0] == 0 and img[i][j][1] == 255 and img[i][j][2] == 0):
            healthy_mask[i][j] = 255

healthy_mask = binary_fill_holes(healthy_mask)

imshow(img_as_ubyte(healthy_mask))
show()

img = img_as_ubyte(imread('saved_data/result/tcc_contour_filed.png'))

result = np.zeros(shape, dtype=int)
result[:,:,2].fill(255)

for i in range(shape[0]):
    for j in range(shape[1]):
        if healthy_mask[i][j]:
            result[i][j][0] = 0
            result[i][j][1] = 255
            result[i][j][2] = 0

for i in range(shape[0]):
    for j in range(shape[1]):
        if img[i][j] == 255:
            result[i][j][0] = 255
            result[i][j][1] = 0
            result[i][j][2] = 0

imsave('saved_data/result/tcc_result_final.png', img_as_ubyte(result))