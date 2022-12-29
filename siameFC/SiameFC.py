import tensorflow as tf

from keras import layers
from keras.models import Model
from keras import backend as K
from keras.layers import (
    Input,
    Lambda,
)
#import backbone.convolutional_alexnet as backbone

#test
import backbone.alexnet_testpb as alexnet

def x_corr_map(x):

    def _translation_match(i):  
        x, z = i[0], i[1]
        x = tf.expand_dims(x, 0)  
        z = tf.expand_dims(z, -1)  
        return tf.nn.conv2d(x, z, strides=[1, 1, 1, 1], padding='VALID', name='translation_match')

    output = tf.map_fn(_translation_match, x, dtype=tf.float32)
    output = tf.squeeze(output, [1, 4])

    return output

def x_corr_layer():
    return Lambda(x_corr_map, output_shape=(17, 17))
    

def siameFc_model(x_shape, z_shape):

    '''
    exemplar = Input(shape=z_shape)
    search = Input(shape=x_shape)
    score_map = Input(shape=(17,17))

    embedding_exemplar = backbone.convolutional_alexnet(exemplar)
    embedding_search = backbone.convolutional_alexnet(search)

    score_map = x_corr_layer()([embedding_search, embedding_exemplar])
    
    outputs = [score_map]
    inputs = [search, exemplar]
    '''
    
    exemplar = Input(shape=z_shape)
    search = Input(shape=x_shape)
    
    #model = Model(inputs=inputs, outputs=outputs)

    alex_net = alexnet.alex_net_layers()


    exemplar_features = alexnet.apply_layers(exemplar, alex_net)
    search_features = alexnet.apply_layers(search, alex_net)
    score_map = x_corr_layer()([search_features, exemplar_features])

    outputs = [score_map]
    inputs = [search, exemplar]

    model = Model(inputs=inputs, outputs=outputs)

    return model


    










