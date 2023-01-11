import tensorflow as tf
import os 
import numpy as np


from keras.utils import image_utils
from siameFC.SiameFC import siameFc_model
from siameFC.loss_of_scoreMap import loss_fn
from siameFC.utils import  load_images, make_ground_th_label, make_bbox_label

tf.compat.v1.disable_eager_execution()


#prameter
Z_SHAPE = (127, 127, 3)
X_SHAPE = (255, 255, 3)

response_size = 17 #score map size
final_stride = 8 

#train Kinds
'''!---- check the train_label ------!'''
train_label = 'gt' #'score' OR 'gt'


#data path 
x_dir = "./sample/VOT19/car1/x/" #number of 742
z_dir = "./sample/VOT19/car1/z/" #number of 1


#img name list
x_name_lsit = os.listdir(x_dir)
z_name_lsit = os.listdir(z_dir)


#make image list
data_size = len(os.listdir(x_dir)) #VOT car1 data_size : 742

x_images = load_images(x_dir, x_name_lsit, 255, data_size, input='x')
z_images = load_images(z_dir, z_name_lsit, 127, data_size, input='z')


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


#!----train start
model = siameFc_model(X_SHAPE,Z_SHAPE, train_label)

#model.summary()

opt = tf.keras.optimizers.legacy.SGD( 
         name='SGD', learning_rate= 0.0001
        )
        
# model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy']) # label : score
model.compile(optimizer=opt, loss='mse', metrics=['accuracy']) #label : gt

#prameter
batch_size = 8
epochs = 50 


model.fit([x_images, z_images], [label], batch_size=batch_size, epochs=epochs)

#save weights
model.save_weights('saved_weights02.h5')

#!----train end