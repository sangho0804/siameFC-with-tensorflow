import tensorflow as tf
import os 
import numpy as np

from keras.utils import image_utils
from siameFC.SiameFC import siameFc_model
from siameFC.losses import loss_fn
from siameFC.utils import  load_images, make_ground_th_label, make_bbox_label

tf.compat.v1.experimental.output_all_intermediates(True)
tf.compat.v1.disable_eager_execution()
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


'''
        일단 정리를 해보자
        
        돌고돌아 input data 가 타당 한가? 이를 먼저 해결해 보자.

        1)
        input label 은 ground_truth 이므로, label size = (batch, 4)
        따라서 output size 는 (None, 4) 가 나와야 한다.

        하지만, 모델 아키텍쳐를 보면 output size = (None, 17, 17)

        따라서 이것에 맞춰서 모델 아키텍쳐를 변경 해줘야 하는것이라 생각한다.

        본 구현의 경우 SiameFC 에서 output 을 Flatten 을 하고 Dense 로 4를 내보내도록 진행했음.

        ** 여기서 activation func 을 무엇을 사용해야 하는지
        ** 모델 아키텍처 변경에서 Dense를 진행하는것이 맞는지

                1 - 1) label input 을 어떻게 해야 하는가?
                        고려해야함.
        
        2) 그렇다면 loss 를 어떻게 해야하는가?
                논문에 명시된 loss 를 사용할 수 있는가?
                        
                        나의 생각으론 사용할 수 없음.

                        본 논문에선 y_true 를 1 또는 -1 을 사용한다.

                
                따라서 mse 혹은 L1 smooth loss 를 이용해보자.

                2-1) metric 을 무엇을 사용해야 하는가?
                        IoU ? AUC ?
                
                그럼에도 학습이 잘 되진 않는다.
                        IoU 를 구현 해봤지만 잘 나오진 않았다. **구현의 문제일듯?
                        AUC 역시 잘 나오진 않았다. **why?
                
                하나 생각해볼것. sigmoid 를 사용하면, output 의 범위는 0 < x < 1 이 될 것인데 이것을 어떻게 tracking 할 수 있는가?
                

                궁금한 것 : 모델 output 은 어떻게 나오는가? 
                *** 이거 진짜 확인 해보자. 시간좀 들여서.


        3) 위의 이유로 score map 학습 방법을 사용할 수 있는가?

        3-1) 일단 load img 가 잘못된것을 확인 하였다. 이것을 고쳐 보자.


'''


#prameter
Z_SHAPE = (127, 127, 3)
X_SHAPE = (255, 255, 3)

response_size = 17 #score map size
final_stride = 8 
radius = 16

#train Kinds
'''!---- check the train_label ------!'''
train_label = 'score' #'score' OR 'gt'
gt_val = 'center'     # bbox is 'corner' OR 'cneter' 

#data path
x_dir = "./sample/VOT19/car1/x/"
z_dir = "./sample/VOT19/car1/z/"


#img name list
x_name_lsit = os.listdir(x_dir)
z_name_lsit = os.listdir(z_dir)


#make image list
data_size = len(os.listdir(x_dir)) #- 19700

#load image 
x_images = load_images(x_dir, x_name_lsit, 255, data_size, input='x')
z_images = load_images(z_dir, z_name_lsit, 127, data_size, input='z')


#data normalization
x_images = x_images / 255.
z_images = z_images / 255.



#ground truth
ground_th_dir = "./sample/VOT19/car1/label/groundtruth.txt"
ground_th = np.loadtxt(ground_th_dir, delimiter=',') 

#for prevent out of memory
#ground_th = ground_th[0:10000,:]


# make label
# label : score OR gt
if train_label == 'score':

        #original image size
        org_img = image_utils.load_img(x_dir + x_name_lsit[0])
        org_img_tensor = image_utils.array_to_img(org_img)
        org_img_size = org_img_tensor.size

        
        label = make_ground_th_label(data_size,final_stride, response_size, ground_th, org_img_size) 

if train_label == 'gt':
        label = make_bbox_label(data_size,ground_th, gt_val)








#!----train start
model = siameFc_model(X_SHAPE,Z_SHAPE, train_label)

#model.summary()

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=1000,
    decay_rate=0.9)
opt = tf.keras.optimizers.legacy.SGD(learning_rate=lr_schedule)



model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy']) # label : score
# model.compile(optimizer=opt, loss='mse', metrics=['accuracy']) #label : gt

#prameter
batch_size = 8
epochs = 1


hist = model.fit([x_images, z_images], [label], batch_size=batch_size, epochs=epochs)

#save weights
model.save_weights('weight_score_test.h5')

#!----train end



import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')

acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')

acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='upper left')

plt.show()