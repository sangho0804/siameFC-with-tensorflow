import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Input,
    Lambda,
)
import siameFC.convolutional_alexnet as alexnet

"""
 modify
    this model have to work type double.
    but Why? -> 

    why correct dim ?
    
"""

def make_score_map(x):
    
    #for correct dim
    # dim : b * h * w * c
    def _translation_match(i):
        x, z = i[0], i[1]
        x = tf.expand_dims(x, 0)  
        z = tf.expand_dims(z, -1)  
        return tf.nn.conv2d(x, z, strides=1, padding='VALID')

    output = tf.map_fn(_translation_match, x, dtype=tf.float32)

    return output


# add Lambda layer (correlation)
def score_map_layer():
    return Lambda(make_score_map, output_shape=(17, 17))
    

def siameFc_model(x_shape, z_shape):

    exemplar = Input(shape=(z_shape))
    search = Input(shape=(x_shape))
    score_map = Input(shape=(17,17))

    embedding_exemplar = alexnet.convolutional_alexnet(exemplar)
    embedding_search = alexnet.convolutional_alexnet(search)

    score_map = score_map_layer()([embedding_search, embedding_exemplar])
    
    outputs = [score_map]
    inputs = [search, exemplar]


    model = Model(inputs, outputs)

    return model


    










