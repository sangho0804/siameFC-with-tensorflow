import tensorflow as tf

from keras import layers
from keras.models import Model
from keras import backend as K
from keras.layers import (
    Input,
    Lambda,
    Conv2D,
    ReLU, 
    MaxPool2D, 
    BatchNormalization
    )
import backbone.convolutional_alexnet as backbone

Z_SHAPE = (127, 127, 3)
X_SHAPE = (255, 255, 3)

class SiameseFC(Model):
    def __init__(self):
        super(SiameseFC, self).__init__()
        self.embedding = backbone()

    def x_corr_map(x):

        def _translation_match(i):  # translation match for one example within a batch
            x, z = i[0], i[1]
            x = tf.expand_dims(x, 0)  # [1, in_height, in_width, in_channels]
            z = tf.expand_dims(z, -1)  # [filter_height, filter_width, in_channels, 1]
            return tf.nn.conv2d(x, z, strides=[1, 1, 1, 1], padding='VALID', name='translation_match')

        output = tf.map_fn(_translation_match, x, dtype=tf.float32)
        output = tf.squeeze(output, [1, 4])  # of shape e.g., [8, 15, 15]

    
        return output

    def x_corr_layer():
        return Lambda(x_corr_map, output_shape=(17, 17))
        


    def Correlation(self, x, z):
        return tf.squeeze(tf.nn.conv2d(x, z, [1,1,1,1], 'VALID'), 0)

    def cross_correlation(inputs):
        x = inputs[0]
        x = tf.reshape(x, [1] + x.shape.as_list())
        z = inputs[1]
        z = tf.reshape(z, z.shape.as_list() + [1])
        return tf.nn.convolution(x, z, padding='VALID', strides=(1,1))




    def model(self, x_shape, z_shape):
        exemplar = Input(shape=z_shape)
        search = Input(shape=x_shape)
        label_input = Input(shape=(17,17))

        embedding_exemplar = backbone.convolutional_alexnet(exemplar).output
        embedding_exemplar = backbone.convolutional_alexnet(search).output



        

        score_map = x_corr_layer()([search_features, exemplar_features])

        return Model(inputs=[exp, src], outputs=self.call([exp, src], False))


        


    







