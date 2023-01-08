import tensorflow as tf
import siameFC.convolutional_alexnet as alexnet

from keras import backend as K
from keras.models import Model
from keras.layers import (
    Input,
    Lambda,
)

#for make correlation score map
def make_score_map(x):
    
    def _translation_match(i):
        x, z = i[0], i[1]
        x = tf.expand_dims(x, 0)   # H * W * C -> b * H * W * C
        z = tf.expand_dims(z, -1)  # H * W * C -> H * W * i_C * o_C

        return tf.nn.conv2d(x, z, strides=[1, 1, 1, 1], padding='VALID')

    output = tf.map_fn(_translation_match, x, dtype=tf.float32) # shape (None * 1 * 17 * 17 * 1)
    output = tf.reshape(output, shape=(-1,17,17))  #reshape (b * 17 * 17)

    return output

# add Lambda layer (correlation)
def score_map_layer():
    return Lambda(lambda x : make_score_map(x), output_shape=(17, 17))


def siameFc_model(x_shape, z_shape):

    exemplar = Input(shape=(z_shape))
    search = Input(shape=(x_shape))
    score_map = Input(shape=(17,17))

    embedding_exemplar = alexnet.convolutional_alexnet(exemplar)
    embedding_search = alexnet.convolutional_alexnet(search)

    score_map = score_map_layer()([embedding_search, embedding_exemplar])
    

    outputs = [score_map]
    inputs = [search, exemplar]

    model = Model(inputs=inputs, outputs=outputs)

    return model


    










