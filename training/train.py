import tensorflow as tf
import custom_loss.loss_of_scoreMap as scoreMap_loss
import siameFC.SiameFC as siameFc
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=Warning)

from data_processing.data import make_label, load_images


if __name__ == '__main__':
        Z_SHAPE = (127, 127, 3)
        X_SHAPE = (255, 255, 3)
        LABEL = (17, 17, 1)

        x_dir = "./sample/"
        z_dir = "./sample/"
        x_images = load_images(x_dir, 255, 0, ".x.jpg")
        z_images = load_images(z_dir, 127, 0, ".z.jpg")

        radius = 16 #hyperparameter

        response_size = 17 #score map size
        response_stride = 8.0 #final stride = 8

        data_size = 1 # yet

        label = make_label(response_size, radius / response_stride)
        labels = np.empty((data_size,) + label.shape)
        labels[:] = label


        model = siameFc.siameFc_model(X_SHAPE,Z_SHAPE)
        model.summary()

        opt = tf.keras.optimizers.SGD(
                momentum=0.9, nesterov=False, name='SGD'
                )

        # what is metrics parameter ?         
        model.compile(optimizer=opt, loss=scoreMap_loss, metrics=['accuracy'])


        batch_size = 1 #yat
        epochs = 1 #yat
        model.fit([x_images, z_images], [labels], batch_size=batch_size, epochs=epochs)