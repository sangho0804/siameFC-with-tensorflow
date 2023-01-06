import numpy as np

from math import ceil
from keras.utils import img_to_array, load_img
from math import sqrt

# for image load & make img list
def load_images(directory, name_list, dimension, n_images):
    """
        directory : image path
        name_list : A list of image names
        dimesion : img target size
                ex) siameFC have to z size : 127 127 3
                    siameFC have to x size : 255 255 3
        n_images : number of image

    """

    img_array = np.empty((n_images, dimension, dimension, 3))
    
    for i in range(0, n_images):
        img = load_img(directory + name_list[i], target_size=(dimension, dimension))
        img_array[i] = img_to_array(img)

    return  img_array



#make ground truth
"""
    make label
    It need to modify.

"""

def make_label(dim, radius):

    label = np.full((dim, dim), -1)
    center = int(dim / 2.0)
    start = center - ceil(radius)
    end = center + ceil(radius)

    for i in range(start, end + 1):
        for j in range(start, end + 1):
            if euclidean_distance(i, j, center, center) <= radius:
                label[i,j] = 1
    return label


def euclidean_distance(x1, y1, x2, y2):
    return sqrt((x1 - x2)**2 + (y1 - y2)**2)