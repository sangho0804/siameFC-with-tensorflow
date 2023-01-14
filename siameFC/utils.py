import numpy as np
import tensorflow as tf

from math import ceil
from math import sqrt

from sklearn.preprocessing import MinMaxScaler

from keras.utils import img_to_array, load_img

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops

# for image load & make img list
def load_images(directory, name_list, dimension, n_images, input='x'):
    """
        directory : image path
        name_list : A list of image names
        dimesion  : img target size
                      ex) z size : 127 127 3
                          x size : 255 255 3
        n_images  : number of image
        input     : x_image = 'x'
                    z_image = 'z'
                    
        return img_array shape = (n_images, dim, dim, 3)
    """

    img_array = np.empty((n_images, dimension, dimension, 3))
    
    if input == 'z': #load z_images 

        img = load_img(directory + name_list[0], target_size=(dimension, dimension))
        img_array[0] = img_to_array(img)
        img_array[:] = img_array[0] 

    else: #load x_images

        for i in range(0, n_images):
            img = load_img(directory + name_list[i], target_size=(dimension, dimension))
            img_array[i] = img_to_array(img)
        
    return  img_array

#make ground truth label processing
def make_ground_th_label(data_size, final_stride, dim, ground_th, org_img_sz):

    '''
        reson of use scale :
        if ground_th_x1 = 320, 
        x1 = || a * final_stride * scale || ( 0 < a < 17 )
        then, 
        if a = 8 ( score_map center)
        x1 = 8 * 8 * scale ≒ 320

        ground th value : x1, y1, x2, y2, x3, y3, x4, y4
        we use (x1, y1), (x3, y3)
        becuase, left_top and right_bottom 

        then, 
        The value in the range is +1

        retrurn : 
        (data_size * 17 * 17) score map list
    '''

    label = np.full((data_size, dim, dim), -1)

    #exampler size 127 * 127
    scale_x = org_img_sz[0] / 127
    scale_y = org_img_sz[1] / 127

    start_x = ground_th[:, 0] / final_stride / scale_x #left top x
    start_y = ground_th[:, 1] / final_stride / scale_y #left top y
    end_x = (ground_th[:, 4] / final_stride / scale_x) + 1 #right dowm x
    end_y = (ground_th[:, 5] / final_stride / scale_y) + 1 #right dowm y

    #within range, set +1
    for i in range(0, data_size):
        label[i, int(start_x[i]) : int(end_x[i]), int(start_y[i]) : int(end_y[i])] = 1

    return label

#make ground truth position label processing
def make_bbox_label(data_size, ground_th):

    '''
        ground th value : x1, y1, x2, y2, x3, y3, x4, y4
        we use (x1, y1), (x3, y3)
        becuase, left_top and right_bottom 

        each value has minmax norm

        return :
        (data_size * 4) bbox list
    '''

    bbox = np.empty((data_size, 4))

    #ground_th (x1, y1), (x3, y3) : left_top, right_bottom 

    bbox[:, 0] = (ground_th[:, 0]) # left_top x 
    bbox[:, 1] = (ground_th[:, 1]) # left_top y
    bbox[:, 2] = (ground_th[:, 4]) # right_bottom x
    bbox[:, 3] = (ground_th[:, 5]) # right_bottom y

    scaler = MinMaxScaler()
    bbox = scaler.fit_transform(bbox)

    return bbox

def IoU_metric(y_true, y_pred):

    tf.keras.metrics.IoU

    '''
        custom metric IoU
        single calss detection model
        : ground_th metric has to IoU

        y_true shape (b_size, 4) || 4 : (left_top_x, left_top_y, right_down_x, right_down_y)
        y_pred shape (b_size, 4) || 4 : model out (b, 4) || 4 : danse(4) 
                                 || pred((left_top_x, left_top_y, right_down_x, right_down_y))

        IoU : intersection area / (predict box area + gt box area - intersection area)

        return : 
        IoU tensor
    
    '''
    x, y =  y_pred.shape
    gt_box_area = (y_true[:, 2] - y_true[:, 0]) * (y_true[:, 3] - y_true[:, 1]) 
    pred_box_area = (y_pred[:, 2] - y_pred[:, 0]) * (y_pred[:, 3] - y_pred[:, 1])

    #intersection

    #position
    inter_lt_x = np.max(y_true[:,0], y_pred[:,0], axis=1)
    inter_lt_y = np.max(y_true[:,1], y_pred[:,1], axis=1)
    inter_rd_x = np.max(y_true[:,2], y_pred[:,2], axis=1)
    inter_rd_y = np.max(y_true[:,3], y_pred[:,3], axis=1)



    return 0


#make label without ground_th
def make_label(dim, radius, data_size):
    
    label = np.full((data_size, dim, dim), -1)
    center = int(dim / 2.0)
    start = center - ceil(radius)
    end = center + ceil(radius)
    
    for k in range(0, data_size-1):
        for i in range(start, end + 1):
            for j in range(start, end + 1):
                if euclidean_distance(i, j, center, center) <= radius:
                    label[k,i,j] = 1
    return label

def euclidean_distance(x1, y1, x2, y2):
    return sqrt((x1 - x2)**2 + (y1 - y2)**2)