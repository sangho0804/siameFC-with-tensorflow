import numpy as np
from math import ceil
from .utill import euclidean_distance, inclusive_range
from keras.utils import img_to_array, load_img


def make_label(dim, radius):

    label = np.full((dim, dim), -1)
    center = int(dim / 2.0)
    start = center - ceil(radius)
    end = center + ceil(radius)

    for i in inclusive_range(start, end):
        for j in inclusive_range(start, end):
            if euclidean_distance(i, j, center, center) <= radius:
                label[i,j] = 1
    return label


def load_images(directory, dimension, n_images, suffix):

    img_array = np.empty((n_images, dimension, dimension, 3))
    
    for i in range(1, n_images + 1):
        img = load_img(directory + str(i) + suffix, target_size=(dimension, dimension))
        img_array[i - 1] = img_to_array(img)
    return  img_array