import tensorflow as tf
import os 
import numpy as np

from sklearn.model_selection import train_test_split
from keras.utils import image_utils
from siameFC.SiameFC import siameFc_model
from siameFC.losses import loss_fn
from siameFC.utils import  load_images, make_ground_th_label, make_bbox_label
import matplotlib.pyplot as plt
import cv2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
tf.compat.v1.disable_eager_execution()


# #prameter
# Z_SHAPE = (127, 127, 3)
# X_SHAPE = (255, 255, 3)

# response_size = 17 #score map size
# final_stride = 8 

# #train Kinds
# '''!---- check the train_label ------!'''
# train_label = 'score' #'score' OR 'gt'


# #data path 
# x_dir = "./sample/VOT19/car1/x/" #number of 742
# z_dir = "./sample/VOT19/car1/z/" #number of 1


# #img name list
# x_name_lsit = os.listdir(x_dir)
# z_name_lsit = os.listdir(z_dir)


# #make image list
# data_size = len(os.listdir(x_dir)) #VOT car1 data_size : 742

# x_images = load_images(x_dir, x_name_lsit, 255, data_size, input='x')
# z_images = load_images(z_dir, z_name_lsit, 127, data_size, input='z')

#246.13, 162.64,    362.69, 168.59,    357.02, 279.72, 240.46, 273.77

# w =  357.02 - 246.13  
# h = 279.72 - 162.64

# x = 4 * (127*127) / ( (3 * w + h) * (w + 3 * h) )

# print(x)
# test = tf.math.sqrt(x)
# sess = tf.compat.v1.Session()
# with sess.as_default():   # or `with sess:` to close on exit
#     assert sess is tf.compat.v1.get_default_session()
#     print(test.eval())


# test1 = 640 * test

# sess = tf.compat.v1.Session()
# with sess.as_default():   # or `with sess:` to close on exit
#     assert sess is tf.compat.v1.get_default_session()
#     print(test1.eval())

CROP_SIZE = (255, 255)

import matplotlib.pyplot as plt
from PIL import Image
import io
box = [246.13,162.64,357.02,279.72]
box_indices = tf.random.uniform(shape=(1,), minval=0, maxval=1, dtype=tf.int32)
# img = Image.open('./sample/VOT19/car1/z/00000001.jpg')
with open('./sample/VOT19/car1/z/00000001.jpg', "rb") as f:
        img = Image.open(io.BytesIO(f.read()))
im2arr = np.array(img)


print(im2arr.shape)

w, h, c = im2arr.shape

x1, y1, x3, y3 = box

center_x , center_y = ((x3+x1)/2, (y3+y1)/2)
center_x, center_y = int(center_x), int(center_y)

print(center_x, center_y)

new_img = np.zeros([255,255,3])


im2arr = im2arr[center_x-int(255/2) : center_x +int(255/2)+1, center_y-int(255/2) : center_y +int(255/2)+1, : ]




# for i in range(0, 3):
#         for j in range(0,255):
                
#                 if j - int(255/2) < 0:
#                         pix_j = j - int(255/2)

#                 if j - int(255/2) == 0:
#                         pix_j = 0

#                 if j - int(255/2) > 0:
#                         pix_j = j + int(255/2)
                
                
#                 for k in range(0,255):
#                         if j - int(255/2) < 0:
#                                 pix_k = k - int(255/2)
#                         if j - int(255/2) == 0:
#                                 pix_k = 0                        
#                         if k - int(255/2) > 0:
#                                 pix_k = k + int(255/2)

#                         if center_x + pix_j > -1 and center_x + pix_j < 481 and center_y + pix_k > -1 and center_y + pix_k < 641:
#                                 org_x = (center_x + pix_j)
#                                 org_y = (center_y + pix_k)
#                                 #print(org_x, org_y)
#                                 #print(org_x, org_y)
#                                 new_img[j,k,i] = im2arr[org_x, org_y,i]






#new_img = new_img.eval(session=tf.compat.v1.Session())
image = tf.keras.utils.array_to_img(im2arr)
plt.figure()
plt.imshow(image)
plt.show()


assert False

import matplotlib.pyplot as plt
from PIL import Image
import io
# img = Image.open('./sample/VOT19/car1/z/00000001.jpg')
with open('./sample/VOT19/car1/z/00000001.jpg', "rb") as f:
        img = Image.open(io.BytesIO(f.read()))
im2arr = np.array(img)
test = tf.image.resize_with_crop_or_pad(im2arr, 255,255)
print(test.shape)
test = test.eval(session=tf.compat.v1.Session())

image = tf.keras.utils.array_to_img(test)
plt.figure()
plt.imshow(image)
plt.show()
assert False

# with sess.as_default():   # or `with sess:` to close on exit
#     assert sess is tf.compat.v1.get_default_session()
#     print(test.eval())

# image = tf.constant(np.arange(1, 28, dtype=np.float32), shape=[3, 3, 3])
# with sess.as_default():   # or `with sess:` to close on exit
#     assert sess is tf.compat.v1.get_default_session()
#     print(image[:,:,0].eval())
# cropped_image = tf.image.crop_to_bounding_box(image, 0, 0, 2, 2)
# with sess.as_default():   # or `with sess:` to close on exit
#     assert sess is tf.compat.v1.get_default_session()
#     print(cropped_image[:,:,0].eval())


# test = tf.image.resize_with_crop_or_pad(x_images[0], )

assert False

#data normalization
x_images = x_images / 255.
z_images = z_images / 255.


#ground truth
ground_th_dir = "./sample/VOT19/car1/label/groundtruth.txt"
ground_th = np.loadtxt(ground_th_dir, delimiter=',') #shape (742,8)



#make label
#label : score OR gt
if train_label == 'score':

        #original image size
        org_img = image_utils.load_img(x_dir + x_name_lsit[0])
        org_img_tensor = image_utils.array_to_img(org_img)
        org_img_size = org_img_tensor.size #VOT img size : 640x480

        
        label = make_ground_th_label(data_size,final_stride, response_size, ground_th, org_img_size) # shape 742 x 17 x 17

if train_label == 'gt':

        label = make_bbox_label(data_size,ground_th) # shape 742 x 4


# Image show

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10, 10) # set figure size

 

plt.imshow(x_images[0])

plt.show()




assert False
#!----train start
model = siameFc_model(X_SHAPE,Z_SHAPE, train_label)

#model.summary()


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=1000,
    decay_rate=0.9)
opt = tf.keras.optimizers.legacy.SGD(learning_rate=lr_schedule)
        
model.compile(optimizer=opt, loss=loss_fn) # label : score
# model.compile(optimizer=opt, loss='mse', metrics=['accuracy']) #label : gt

batch_size = 8
epochs = 50 

model.load_weights('weight_gt.h5')


#!----test tracking

y_evaluate = model.evaluate([x_images, z_images], label, batch_size=8)

print("check loss : ", y_evaluate)

predict = model.predict([x_images, z_images])




# !---------------------------IoU test start

# def calculate_iou(y_true, y_pred):
    
    
#     """
#     Input:
#     Keras provides the input as numpy arrays with shape (batch_size, num_columns).
    
#     Arguments:
#     y_true -- first box, numpy array with format [x, y, width, height, conf_score]
#     y_pred -- second box, numpy array with format [x, y, width, height, conf_score]
#     x any y are the coordinates of the top left corner of each box.
    
#     Output: IoU of type float32. (This is a ratio. Max is 1. Min is 0.)
    
#     """

    
#     results = []
    
#     for i in range(0,y_true.shape[0]):
    
#         # set the types so we are sure what type we are using
#         y_true = y_true.astype(np.float32)
#         y_pred = y_pred.astype(np.float32)


#         # boxTrue
#         x_boxTrue_tleft = y_true[i,0]  # numpy index selection
#         y_boxTrue_tleft = y_true[i,1]
#         boxTrue_width = y_true[i,2]
#         boxTrue_height = y_true[i,3]
#         area_boxTrue = (boxTrue_width * boxTrue_height)

#         # boxPred
#         x_boxPred_tleft = y_pred[i,0]
#         y_boxPred_tleft = y_pred[i,1]
#         boxPred_width = y_pred[i,2]
#         boxPred_height = y_pred[i,3]
#         area_boxPred = (boxPred_width * boxPred_height)

#         print('true :', area_boxPred, area_boxTrue)
#         # calculate the bottom right coordinates for boxTrue and boxPred

#         # boxTrue
#         x_boxTrue_br = x_boxTrue_tleft + boxTrue_width
#         y_boxTrue_br = y_boxTrue_tleft + boxTrue_height # Version 2 revision

#         # boxPred
#         x_boxPred_br = x_boxPred_tleft + boxPred_width
#         y_boxPred_br = y_boxPred_tleft + boxPred_height # Version 2 revision


#         # calculate the top left and bottom right coordinates for the intersection box, boxInt

#         # boxInt - top left coords
#         x_boxInt_tleft = np.max([x_boxTrue_tleft,x_boxPred_tleft])
#         y_boxInt_tleft = np.max([y_boxTrue_tleft,y_boxPred_tleft]) # Version 2 revision

#         # boxInt - bottom right coords
#         x_boxInt_br = np.min([x_boxTrue_br,x_boxPred_br])
#         y_boxInt_br = np.min([y_boxTrue_br,y_boxPred_br]) 

#         # Calculate the area of boxInt, i.e. the area of the intersection 
#         # between boxTrue and boxPred.
#         # The np.max() function forces the intersection area to 0 if the boxes don't overlap.
        
        
#         # Version 2 revision
#         area_of_intersection = \
#         np.max([0,(x_boxInt_br - x_boxInt_tleft)]) * np.max([0,(y_boxInt_br - y_boxInt_tleft)])

#         iou = area_of_intersection / ((area_boxTrue + area_boxPred) - area_of_intersection)


#         # This must match the type used in py_func
#         iou = iou.astype(np.float32)
        
#         # append the result to a list at the end of each loop
#         results.append(iou)
    
#     # return the mean IoU score for the batch
#     return np.mean(results)
# y_true = np.array([[1,5,2,3,0.5], [1,5,2,3,0.5], [1,5,2,3,0.5]])
# y_pred = np.array([[2,4,3,3,0.7], [2,4,3,3,0.7], [2,4,3,3,0.7]])

# print(y_true.shape)
# result = calculate_iou(y_true, y_pred)

# print(result)
# print(type(result))

# y_true = np.array([[1,5,3,8], [1,5,3,8], [1,5,3,8]])
# y_pred = np.array([[2,4,5,7], [2,4,5,7], [2,4,5,7]])

# results =[]
# for i in range(0, y_true.shape[0]):
#         y_true = y_true.astype(np.float32)
#         y_pred = y_pred.astype(np.float32)

#         # set the types so we are sure what type we are using
#         true_left_x = y_true[i,0] * (y_true[i, 2]- y_true[i, 0]) #2
#         true_left_y = y_true[i,1] * (y_true[i, 3]- y_true[i, 1])
#         true_right_x = y_true[i,2] * (y_true[i, 2]- y_true[i, 0]) #6
#         true_right_y = y_true[i,3] * (y_true[i, 3]- y_true[i, 1])

#         print("test1: ",true_left_x, true_left_y, true_right_x, true_right_y)

#         pred_left_x = y_pred[i,0] * (y_pred[i, 2]- y_pred[i, 0]) # 6
#         pred_left_y = y_pred[i,1] * (y_pred[i, 3]- y_pred[i, 1])
#         pred_right_x = y_pred[i,2] * (y_pred[i, 2]- y_pred[i, 0]) # 15
#         pred_right_y = y_pred[i,3] * (y_pred[i, 3]- y_pred[i, 1])
#         print('정체를 밝혀라',pred_right_x)

#         gt_box_area = ( (true_right_x) - true_left_x) * ( (true_right_y) - true_left_y) 
#         pred_box_area = ( (pred_right_x) - pred_left_x) * ( (pred_right_y) - pred_left_y)
#         print(gt_box_area, pred_box_area)

#         # calculate the top left and bottom right coordinates for the intersection box, boxInt

#         # boxInt - top left coords
#         x_boxInt_tleft = np.max([ true_left_x, pred_left_x ])
#         y_boxInt_tleft = np.max([true_left_y, pred_left_y ]) # Version 2 revision

#         # boxInt - bottom right coords
#         x_boxInt_br = np.min([true_right_x,pred_right_x ]) # 15 
#         y_boxInt_br = np.min([true_right_y,pred_right_y]) 

#         # Calculate the area of boxInt, i.e. the area of the intersection 
#         # between boxTrue and boxPred.
#         # The np.max() function forces the intersection area to 0 if the boxes don't overlap.


#         # Version 2 revision
#         area_of_intersection = np.max([0,(x_boxInt_br - x_boxInt_tleft)]) * np.max([0,(y_boxInt_br - y_boxInt_tleft)])
#         print(x_boxInt_br, x_boxInt_tleft)
#         print('ab',np.max([0,(x_boxInt_br - x_boxInt_tleft)]))
#         print('aab',np.max([0,(y_boxInt_br - y_boxInt_tleft)]))
#         print('뭐야',area_of_intersection)

#         iou = area_of_intersection / ((gt_box_area + pred_box_area) - area_of_intersection)


#         # This must match the type used in py_func
#         iou = iou.astype(np.float32)

#         # append the result to a list at the end of each loop
#         results.append(iou)

# # return the mean IoU score for the batch
# print('test : ',  np.mean(results))

# !---------------------------IoU test end


# !---------------------------loss test start
# #logistic loss
# def logistic_fn(labels=None,logits=None):

#     #convert tensor
#     logits = ops.convert_to_tensor(logits, name="logits")
#     #labels = tf.cast(logits, dtype=logits.dtype)
#     labels = ops.convert_to_tensor(labels, name="labels", dtype=logits.dtype)

#     # print("logits")
#     # sess = tf.compat.v1.Session()
#     # with sess.as_default():   # or `with sess:` to close on exit
#     #     assert sess is tf.compat.v1.get_default_session()
#     #     print(logits.eval())
#     # print("labels")
#     # with sess.as_default():   # or `with sess:` to close on exit
#     #     assert sess is tf.compat.v1.get_default_session()
#     #     print(labels.eval())        

#     #zero tensor
#     #for compared label value
    
#     zeros = array_ops.zeros_like(labels, dtype=logits.dtype)
#     # print("zeros")
#     # with sess.as_default():   # or `with sess:` to close on exit
#     #     assert sess is tf.compat.v1.get_default_session()
#     #     print(zeros.eval())     

#     #If y_true value(+1) >= 0 --> true, 
#     #Else value is  (-1) then false.
     
#     cond = (labels > zeros)
#     # print("cond")
#     # with sess.as_default():   # or `with sess:` to close on exit
#     #     assert sess is tf.compat.v1.get_default_session()
#     #     print(cond.eval()) 


#     #used log(-yx) frame 
#     # If cond (true) then -x, 
#     # Else cond(False) then x 
#     neg_abs_logits = array_ops.where(cond, -logits, logits)
#     # print("cond")
#     # with sess.as_default():   # or `with sess:` to close on exit
#     #     assert sess is tf.compat.v1.get_default_session()
#     #     print(neg_abs_logits.eval()) 

#     return math_ops.log1p(math_ops.exp(neg_abs_logits))


# #total loss function
# def loss_fn(y_true, y_pred):
    
#     #use logistic_fn
#     logistic = logistic_fn(labels=y_true, logits=y_pred)

#     print("logistic")
#     sess = tf.compat.v1.Session()
#     with sess.as_default():   # or `with sess:` to close on exit
#         assert sess is tf.compat.v1.get_default_session()
#         print(logistic.eval())     

#     loss = tf.reduce_sum(logistic, axis=[1,2])

#     print("loss sum")
#     with sess.as_default():   # or `with sess:` to close on exit
#         assert sess is tf.compat.v1.get_default_session()
#         print(loss.eval())  

#     loss = tf.reduce_mean(loss)

#     print("loss mean")
#     with sess.as_default():   # or `with sess:` to close on exit
#         assert sess is tf.compat.v1.get_default_session()
#         print(loss.eval())      
#     return loss

# tt = tf.constant(value=200, shape=(2,17,17), dtype=tf.float32)

# # sess = tf.compat.v1.Session()
# # with sess.as_default():   # or `with sess:` to close on exit
# #     assert sess is tf.compat.v1.get_default_session()
# #     print(tt.eval())

# test = loss_fn(label[0:2], tt)
# # sess = tf.compat.v1.Session()
# # with sess.as_default():   # or `with sess:` to close on exit
# #     assert sess is tf.compat.v1.get_default_session()
# #     print(test.eval())

# #     print(test.shape)

# !---------------------------loss test end


# !--------------------------- tracking 확인 보류용

# b, w, h = predict.shape

# def tracking_object(predict):

#     b, w, h = predict.shape
#     length = w * h
#     center = []
#     response = tf.reshape(predict, [-1, length])

#     ind_max = tf.argmax(response, 1)
#     print("\n\n test")
# #     sess = tf.compat.v1.Session()
# #     with sess.as_default():   # or `with sess:` to close on exit
# #         assert sess is tf.compat.v1.get_default_session()
# #         print(ind_max.eval())



    

#     return ind_max

# test = tracking_object(predict)
# print("\n\n\n test")

# ind_test = []

# for i in range(0, b):
#         for j in range(0, w):
#                 for k in range(0, h):
#                         if test[i] == predict[i,j,k]:
#                                 ind_test.append([i,j])

# print(ind_test)

# #test
# x , y = ind_test[0]
# scale_x = 640 / 127
# scale_y = 480 / 127
# x = x * 8 * scale_x
# y = y * 8 * scale_y

# cv2_image = cv2.imread('./sample/VOT19/car1/x/00000001.jpg', cv2.IMREAD_COLOR)
# drawing_image = cv2_image.copy()
# cv2.circle(drawing_image, (x, y), 10, (0, 0, 255), 3)

# def img_show(title='image', img=None, figsize=(8 ,5)):
#     plt.figure(figsize=figsize)
 
#     if type(img) == list:
#         if type(title) == list:
#             titles = title
#         else:
#             titles = []
 
#             for i in range(len(img)):
#                 titles.append(title)
 
#         for i in range(len(img)):
#             if len(img[i].shape) <= 2:
#                 rgbImg = cv2.cvtColor(img[i], cv2.COLOR_GRAY2RGB)
#             else:
#                 rgbImg = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)
 
#             plt.subplot(1, len(img), i + 1), plt.imshow(rgbImg)
#             plt.title(titles[i])
#             plt.xticks([]), plt.yticks([])
 
#         plt.show()
#     else:
#         if len(img.shape) < 3:
#             rgbImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#         else:
#             rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
#         plt.imshow(rgbImg)
#         plt.title(title)
#         plt.xticks([]), plt.yticks([])
#         plt.show()


# img_show(["Original", "Drawing"], [cv2_image, drawing_image])

# !---------------------------tracking 확인 보류용