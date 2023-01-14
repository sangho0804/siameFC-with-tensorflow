import tensorflow as tf
import os 
import numpy as np

from keras.utils import image_utils
from siameFC.SiameFC import siameFc_model
from siameFC.loss_of_scoreMap import loss_fn
from siameFC.utils import  load_images, make_ground_th_label, make_bbox_label, make_label

tf.compat.v1.experimental.output_all_intermediates(True)
tf.compat.v1.disable_eager_execution()
'''
        score map train 보류 **

        일단 metric : IoU 를 사용해봐야 함 을 먼저 해야 하지 않을까?

        learning rate update : ok

        data set을 늘려보자

        1) epoch 수를 300으로 늘려보자. - 의미없음

'''


#prameter
Z_SHAPE = (127, 127, 3)
X_SHAPE = (255, 255, 3)

response_size = 17 #score map size
final_stride = 8 
radius = 16

#train Kinds
'''!---- check the train_label ------!'''
train_label = 'gt' #'score' OR 'gt'


#data path 
x_dir = "./sample/VOT19/liverRun/x/"
z_dir = "./sample/VOT19/liverRun/z/"


#img name list
x_name_lsit = os.listdir(x_dir)
z_name_lsit = os.listdir(z_dir)


#make image list
data_size = len(os.listdir(x_dir)) 

x_images = load_images(x_dir, x_name_lsit, 255, data_size, input='x')
z_images = load_images(z_dir, z_name_lsit, 127, data_size, input='z')


#data normalization
x_images = x_images / 255.
z_images = z_images / 255.


#ground truth
ground_th_dir = "./sample/VOT19/liverRun/label/groundtruth.txt"
ground_th = np.loadtxt(ground_th_dir, delimiter=',') 

print(ground_th)

assert False


#make label
#label : score OR gt
if train_label == 'score':

        #original image size
        org_img = image_utils.load_img(x_dir + x_name_lsit[0])
        org_img_tensor = image_utils.array_to_img(org_img)
        org_img_size = org_img_tensor.size

        
        label = make_ground_th_label(data_size,final_stride, response_size, ground_th, org_img_size) 

if train_label == 'gt':

        label = make_bbox_label(data_size,ground_th)


assert False
#!----train start
model = siameFc_model(X_SHAPE,Z_SHAPE, train_label)

#model.summary()

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)
opt = tf.keras.optimizers.legacy.SGD(learning_rate=lr_schedule)




# model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy']) # label : score
model.compile(optimizer=opt, loss='mse', metrics=[tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0])]) #label : gt

#prameter
batch_size = 8
epochs = 50


test_model = model.fit([x_images, z_images], [label], batch_size=batch_size, epochs=epochs)

#save weights
model.save_weights('score_map_weight_gt.h5')

#!----train end
