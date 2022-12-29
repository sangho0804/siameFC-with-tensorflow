from keras.layers import (
    Conv2D,
    MaxPool2D, 
    BatchNormalization
    )

def conv_layer(filters, kernel_dim, stride_len):
    return [Conv2D(filters, kernel_dim, strides=stride_len,
                  padding='valid', activation='relu', kernel_initializer='glorot_normal')]

def conv_block(filters, kernel_dim, stride_len):
    batch_norm = [BatchNormalization(axis=3)]
    return conv_layer(filters, kernel_dim, stride_len) + batch_norm

def max_pool():
    return [MaxPool2D(pool_size=3, strides=2, padding='valid')]

def alex_net_layers():
    layers = []
    layers += conv_block(48, 11, 2)
    layers += max_pool()
    layers += conv_block(128, 5, 1)
    layers += max_pool()
    layers += conv_block(48, 3, 1)
    layers += conv_block(48, 3, 1)
    layers += [Conv2D(32, 3, strides=1, padding='valid', kernel_initializer='glorot_normal')]
    return layers

def apply_layers(x, layers):
    out = x
    for layer in layers:
        out = layer(out)
    return out
