import tensorflow as tf
import os 
import numpy as np
import cv2

from keras.utils import image_utils
from siameFC.SiameFC import siameFc_model
from siameFC.loss_of_scoreMap import loss_of_scoreMap
from siameFC.utils import  load_images, make_ground_th_label


Z_SHAPE = (127, 127, 3)
X_SHAPE = (255, 255, 3)

#data path 
x_dir = "./sample/VOT19/car1/x/" #number of 742
z_dir = "./sample/VOT19/car1/z/" #number of 1

#img name list
x_name_lsit = os.listdir(x_dir)
z_name_lsit = os.listdir(z_dir)

#original image size
org_img = image_utils.load_img(x_dir + x_name_lsit[0])
org_img_tensor = image_utils.array_to_img(org_img)
org_img_size = org_img_tensor.size #VOT img size : 640x480


#make image list
data_size = len(os.listdir(x_dir)) #VOT car1 data_size : 742

x_images = load_images(x_dir, x_name_lsit, 255, data_size, input='x')
z_images = load_images(z_dir, z_name_lsit, 127, data_size, input='z')
                

#ground truth
ground_th_dir = "./sample/VOT19/car1/label/groundtruth.txt"
ground_th = np.loadtxt(ground_th_dir, delimiter=',') #shape (742,8)


# #gt 를 17 17 로 맞춰줘야 한다.
# #식을 보자면 sc map 에서 나온 좌표를 a,b 라고 하자
# #그럼 a*8*5 = gt x 좌표가 될것이다. 라고 가정해 보자
# #8은 final stride
# #5 인 이유, VOT car data shape 이 640 480 이기 때문
# #즉 if, sc map 에서 중앙 값 8,8 -> 64,64 가 되고 *5 하면 320
# #이것은 gt 값과 유사 모든건 가정임

# #하지만 우리는 17 17 label 의 1과 -1로 채워야 한다!

# scale_x = org_img_size[0] / 127
# scale_y = org_img_size[1] / 127

# start_x = (ground_th[:, 0] / 8 / scale_x)
# # start_y = int(ground_th[:, 1] / 8 / scale_y)
# # end_x = int(ground_th[:, 4] / 8 / scale_x)
# # end_y = int(ground_th[:, 5] / 8 / scale_y)
# start_x = [int(i) for i in start_x]
# print(start_x[0])
# # print(start_y)
# # print(end_x)
# # print(end_y)

# #좌표 설정

# #slicing


response_size = 17 #score map size
final_stride = 8 
label = make_ground_th_label(data_size, final_stride, response_size, ground_th, org_img_size, 'VOT')

print(label.shape)

model = siameFc_model(X_SHAPE,Z_SHAPE)

#model.summary()


opt = tf.keras.optimizers.SGD(
        momentum=0.9, nesterov=False, name='SGD'
        )


        
model.compile(optimizer=opt, loss=loss_of_scoreMap, metrics=['accuracy'])


batch_size = 8 
epochs = 50 
model.fit([x_images, z_images], [label], batch_size=batch_size, epochs=epochs)

