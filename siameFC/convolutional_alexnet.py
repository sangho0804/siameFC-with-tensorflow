from keras.layers import (
    Conv2D,
    ReLU, 
    MaxPool2D, 
    BatchNormalization
    )

def AlexConv2d(x, filters, size, strides, padding='valid', group=1, conv5='false'):
    
    x = Conv2D(filters=filters, kernel_size=size,
                strides=strides, padding=padding, groups=group)(x)


    x = BatchNormalization()(x)

    #convloultional Alexnet has not ReLu at conv5
    if conv5 != 'true':
        x = ReLU()(x)

    return x


def convolutional_alexnet(input):
    """
        alexnet has parallel progess
        so, we need to group Conv2D (ex : group = 2)
        Refer to the architecture of the paper
    """

    net = input

    #layer 1
    net = AlexConv2d(net, 96, 11, 2)
    net = MaxPool2D(pool_size=3, strides=2, padding='valid')(net)

    #layer 2
    net = AlexConv2d(net, 256, 5, 1, group=2)
    net = MaxPool2D(pool_size=3, strides=2, padding='valid')(net)

    #layer 3
    net = AlexConv2d(net, 384, 3, 1, group=2)

    #layer 4
    net = AlexConv2d(net, 384, 3, 1, group=2)

    #layer 5
    net = AlexConv2d(net, 256, 3, 1, group=2, conv5='true')

    return net


            






















