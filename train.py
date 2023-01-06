import tensorflow as tf
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

if __name__ == '__main__':
        Z_SHAPE = (127, 127, 3)
        X_SHAPE = (255, 255, 3)

        #data path 
        x_dir = "./sample/VOT19/car1/x/"
        z_dir = "./sample/VOT19/car1/z/"

        #make image list
        x_images = load_images(x_dir, 255, 0, ".x.jpg")
        z_images = load_images(z_dir, 127, 0, ".z.jpg")

        
        #test
        # #use ground truth
        # grund_th_dir = "./sample/VOT19/car1/label/groundtruth.txt"


        # # radius = 16 #hyperparameter R

        # # response_size = 17 #score map size
        # # response_stride = 8.0 #final stride = 8

        # # data_size = 1 # yet

        # # label = make_label(response_size, radius / response_stride)
        # # labels = np.empty((data_size,) + label.shape)
        # # labels[:] = label


        # model = siameFc_model(X_SHAPE,Z_SHAPE)
        # model.summary()
        
        
        # opt = tf.keras.optimizers.SGD(
        #         momentum=0.9, nesterov=False, name='SGD'
        #         )
 

             
        # model.compile(optimizer=opt, loss=loss_of_scoreMap, metrics=['accuracy'])


        # batch_size = 1 #yat
        # epochs = 1 #yat
        # model.fit([x_images, z_images], [labels], batch_size=batch_size, epochs=epochs)