import numpy as np
from skimage import img_as_ubyte

def extract_superpixel(args):
    original = args[0]
    segments = args[1]
    value = args[2]
    #calculate the segments image shape and create an empty array
    shape = segments.shape
    coordinates = []
    
    #save all coordinates of the wanted superpixel
    for i in range(shape[0]):
        for j in range(shape[1]):
            if segments[i][j] == value:
                coordinates.append([i, j])

    #transform the list into ndarray in order to use some numpy built-in functions
    coordinates = np.array(coordinates)

    #save the max and min coordinates of width and heigth of the superpixel container box
    box_coordinates = [max(coordinates[:,0]), min(coordinates[:,0]), max(coordinates[:,1]), min(coordinates[:,1])]

    #calculate the shape of the box, based on the max and min width and height coordinates calculated before
    #it is a tridimensional shape because of the rgb model from original image
    box_shape = (box_coordinates[0] - box_coordinates[1] + 1, box_coordinates[2] - box_coordinates[3] + 1, 3)

    #create a box based on the calculeted shape and fill with a constant, in this case -1
    box = np.zeros(box_shape, dtype=int)
    #box.fill(-1)

    #calculate the box corresponding zero on the original image
    zero = [box_coordinates[1], box_coordinates[3]]

    #for each coordinate of the wanted superpixel, add the RGB value into the corresponding position on the box
    for i in coordinates:
        box_point = i - zero
        box[box_point[0], box_point[1]] = original[i[0]][i[1]]
    
    return img_as_ubyte(box)
