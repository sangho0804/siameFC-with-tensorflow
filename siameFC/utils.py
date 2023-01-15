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
def make_bbox_label(data_size, ground_th, gt_val="corner"):

    '''
        ground th value : x1, y1, x2, y2, x3, y3, x4, y4
        we use (x1, y1), (x3, y3)
        becuase, left_top and right_bottom 

        each value has minmax norm

        return :
        (data_size * 4) bbox list
    '''

    bbox = np.empty((data_size, 4))


    if gt_val == 'corner' :
        #ground_th (x1, y1), (x3, y3) : left_top, right_bottom 
        for i in range(0, data_size):

            bbox[i, 0] = float(ground_th[i, 0]) / (ground_th[i, 4]- ground_th[i, 0]) # left_top x 
            bbox[i, 1] = float(ground_th[i, 1]) / (ground_th[i, 5]- ground_th[i, 1]) # left_top y
            bbox[i, 2] = float(ground_th[i, 4]) / (ground_th[i, 4]- ground_th[i, 0]) # right_bottom x
            bbox[i, 3] = float(ground_th[i, 5]) / (ground_th[i, 5]- ground_th[i, 1]) # right_bottom y

    if gt_val == 'center':
        #ground_th (x,y,w,h)
        for i in range(0, data_size):

            bbox[i, 0] = float(ground_th[i,0] - (ground_th[i,2] / 2)) / ground_th[i,2] # left_top x
            bbox[i, 1] = float(ground_th[i,1] - (ground_th[i,3] / 2)) / ground_th[i,3]# left_top y
            bbox[i, 2] = float(ground_th[i,0] + (ground_th[i,2] / 2)) / ground_th[i,2]# right_bottom x
            bbox[i, 3] = float(ground_th[i,1] + (ground_th[i,3] / 2)) / ground_th[i,3]# right_bottom y

    # scaler = MinMaxScaler()
    # bbox = scaler.fit_transform(bbox)

    return bbox

def calculate_iou(y_true, y_pred):
    
    
    """
    Input:
    Keras provides the input as numpy arrays with shape (batch_size, num_columns).
    
    Arguments:
    y_true -- first box, numpy array with format [x, y, width, height, conf_score]
    y_pred -- second box, numpy array with format [x, y, width, height, conf_score]
    x any y are the coordinates of the top left corner of each box.
    
    Output: IoU of type float32. (This is a ratio. Max is 1. Min is 0.)
    
    """
    results = []
    
    for i in range(0, y_true.shape[0]):
    
        # set the types so we are sure what type we are using
        true_left_x = y_true[i,0] #* (y_true[i, 2]- y_true[i, 0])
        true_left_y = y_true[i,1] # (y_true[i, 3]- y_true[i, 1])
        true_right_x = y_true[i,2] #* (y_true[i, 2]- y_true[i, 0])
        true_right_y = y_true[i,3] #* (y_true[i, 3]- y_true[i, 1])

        pred_left_x = y_pred[i,0] #* (y_pred[i, 2]- y_pred[i, 0])
        pred_left_y = y_pred[i,1] #* (y_pred[i, 3]- y_pred[i, 1])
        pred_right_x = y_pred[i,2] #* (y_pred[i, 2]- y_pred[i, 0])
        pred_right_y = y_pred[i,3] #* (y_pred[i, 3]- y_pred[i, 1])

        print(pred_left_x,pred_left_y,pred_right_x,pred_right_y)
        gt_box_area = ( (true_right_x) - true_left_x) * ( (true_right_y) - true_left_y) 
        pred_box_area = ( (pred_left_y) - pred_left_x) * ( (pred_right_y) - pred_right_x)
        print("gt :",gt_box_area)
        print("pred:", pred_box_area)
        # calculate the top left and bottom right coordinates for the intersection box, boxInt

        # boxInt - top left coords
        x_boxInt_tleft = np.max([ true_left_x, pred_left_x ])
        y_boxInt_tleft = np.max([true_left_y, pred_left_y ]) # Version 2 revision

        # boxInt - bottom right coords
        x_boxInt_br = np.min([true_right_x,pred_right_x ])
        y_boxInt_br = np.min([true_right_y,pred_right_y]) 

        # Calculate the area of boxInt, i.e. the area of the intersection 
        # between boxTrue and boxPred.
        # The np.max() function forces the intersection area to 0 if the boxes don't overlap.
        
        
        # Version 2 revision
        area_of_intersection = np.max([0,(x_boxInt_br - x_boxInt_tleft)]) * np.max([0,(y_boxInt_br - y_boxInt_tleft)])

        iou = area_of_intersection / ((gt_box_area + pred_box_area) - area_of_intersection)


        # This must match the type used in py_func
        iou = iou.astype(np.float32)
        
        # append the result to a list at the end of each loop
        results.append(iou)
    
    # return the mean IoU score for the batch
    return np.mean(results)


def IoU(y_true, y_pred):
    
    # Note: the type float32 is very important. It must be the same type as the output from
    # the python function above or you too may spend many late night hours 
    # trying to debug and almost give up.
    
    iou = tf.py_function(calculate_iou, [y_true, y_pred], tf.float32)

    return iou


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