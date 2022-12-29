import tensorflow as tf
import loss.custom_loss as scoreMap_loss
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
    x_images = load_images(x_dir, 255, 0, ".x.jpg", normalize_images=True)
    z_images = load_images(z_dir, 127, 0, ".z.jpg", normalize_images=True)

    positive_label_pixel_radius = 16 # distance from center of target patch still considered a 'positive' match

    response_size = 17
    response_stride = 8.0

    data_size = 1 # yet

    label = make_label(response_size, positive_label_pixel_radius / response_stride)
    labels = np.empty((data_size,) + label.shape)
    labels[:] = label


    model = siameFc.siameFc_model(X_SHAPE,Z_SHAPE)
    model.summary()
    
    opt = tf.keras.optimizers.SGD(
            momentum=0.9, nesterov=False, name='SGD'
            )

    model.compile(optimizer=opt, loss=scoreMap_loss, metrics=['accuracy'])


    batch_size = 1 #yat
    epochs = 1 #yat
    model.fit([x_images, z_images], [labels], batch_size=batch_size, epochs=epochs)