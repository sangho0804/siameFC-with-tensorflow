import tensorflow as tf
import siameFC.convolutional_alexnet as alexnet

from keras import backend as K
from keras.models import Model
from keras.layers import (
    Input,
    Lambda,
    Dense,
    Flatten
)

#for make correlation score map
def make_score_map(x):
    
    def _translation_match(x,z):

        x = tf.expand_dims(x, 0)   # H * W * C -> b * H * W * C
        z = tf.expand_dims(z, -1)  # H * W * C -> H * W * i_C * o_C
    
        return tf.nn.conv2d(x, z, strides=[1, 1, 1, 1], padding='VALID', name='translation_match')

    output = tf.map_fn(lambda x : _translation_match(x[0], x[1]), x, dtype=tf.float32) # shape (None * 1 * 17 * 17 * 1)
    output = tf.squeeze(output, [1, 4]) 

    return output

# add Lambda layer (correlation)
def score_map_layer():
    return Lambda(lambda x : make_score_map(x), output_shape=(17, 17))

def siameFc_model(x_shape, z_shape, train='score'):

    exemplar = Input(shape=(z_shape))
    search = Input(shape=(x_shape))
    score_map = Input(shape=(17,17))

    embedding_exemplar = alexnet.convolutional_alexnet(exemplar)
    embedding_search = alexnet.convolutional_alexnet(search)

    score_map = score_map_layer()([embedding_search, embedding_exemplar])
   

    #score map learning
    outputs = [score_map]
    inputs = [search, exemplar]
    
    #If train bbox, outputs shape = (batch, 4)
    #output =  left top, right down = (x1,y1,x3,y3)
    if train =='gt':
        #position
        #이거 맞나?
        flatten = Flatten()(score_map)
        # gt_position = Dense(32, activation="relu")(flatten)
        gt_position = Dense(4, activation='sigmoid')(gt_position)
        outputs = [gt_position]


    model = Model(inputs=inputs, outputs=outputs)

    return model


    










