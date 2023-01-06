import tensorflow as tf
import os 
import numpy as np

from siameFC.SiameFC import siameFc_model
from siameFC.loss_of_scoreMap import loss_of_scoreMap
from siameFC.utils import  load_images




"""
        zip 으로 묶어서 list 로 
        dic 형태로 만들어서
        json 파일로

        a 인풋 경로 b 인풋경로 정답
        3개 로 

"""

Z_SHAPE = (127, 127, 3)
X_SHAPE = (255, 255, 3)

#data path 
x_dir = "./sample/VOT19/car1/x/" #number of 742
z_dir = "./sample/VOT19/car1/z/" #number of 1

#img name list
x_name_lsit = os.listdir(x_dir)
z_name_lsit = os.listdir(z_dir)

#make image list
x_images = load_images(x_dir, x_name_lsit, 255, len(os.listdir(x_dir)))
z_images = load_images(z_dir, z_name_lsit, 127, len(os.listdir(z_dir)))

train_set = [x_images, z_images]

        
#test
#use ground truth
grund_th_dir = "./sample/VOT19/car1/label/groundtruth.txt"
label = np.loadtxt(grund_th_dir, delimiter=',', dtype=np.float32)

model = siameFc_model(X_SHAPE,Z_SHAPE)
#model.summary()


opt = tf.keras.optimizers.SGD(
        momentum=0.9, nesterov=False, name='SGD'
        )


        
model.compile(optimizer=opt, loss=loss_of_scoreMap, metrics=['accuracy'])


batch_size = 8 
epochs = 50 
model.fit(train_set, label, batch_size=batch_size, epochs=epochs)

