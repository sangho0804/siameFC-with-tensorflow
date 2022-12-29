import tensorflow as tf

from keras.layers import (
    Conv2D,
    ReLU, 
    MaxPool2D, 
    BatchNormalization
    )

def AlexConv2d(x, filters, size, strides, batch_norm=True, padding='valid', name=""):

    x = Conv2D(filters=filters, kernel_size=size,
               strides=strides, padding=padding,
               use_bias=not batch_norm)(x)

    x = BatchNormalization()(x)

    #convloultional Alexnet has not ReLu at conv5
    if name !='conv5':
        x = ReLU()(x)

    return x


def convolutional_alexnet(input):

    net = input

    #layer 1
    net = AlexConv2d(net, 96, 11, 2, 'conv1')
    net = MaxPool2D(pool_size=3, strides=2, padding='valid')

    #layer 2
    b1, b2 = tf.split(net, 2, 3)
    b1 = AlexConv2d(b1, 128, 5, 1, 'conv2')
    b2 = AlexConv2d(b2, 128, 5, 1, 'conv2')
    net = tf.concat([b1, b2], 3)
    net = MaxPool2D(pool_size=3, strides=2, padding='valid')

    #layer 3
    net = AlexConv2d(net, 384, 3, 1, 'conv3')

    #layer 4
    b1, b2 = tf.split(net, 2, 3)
    b1 = AlexConv2d(b1, 192, 3, 1, 'conv2')
    b2 = AlexConv2d(b2, 192, 3, 1, 'conv2')
    net = tf.concat([b1, b2], 3)
    net = MaxPool2D(pool_size=3, strides=2, padding='valid')

    #layer 5
    b1, b2 = tf.split(net, 2, 3)
    b1 = AlexConv2d(b1, 128, 3, 1, 'conv2')
    b2 = AlexConv2d(b2, 128, 3, 1, 'conv2')
    net = tf.concat([b1, b2], 3)

    output = net

    return output


    # def __init__(self, input_shape):
    #     self.input_shape = input_shape

    #     super(convolutional_alexnet, self).__init__()
    #     #layer 1
    #     self.conv1 = Conv2D(96, 11, strides=2, padding='valid')
    #     self.bn1 = BatchNormalization(trainable=False)
    #     self.relu1 = ReLU()
        
    #     self.pool1 = MaxPool2D(pool_size=3, strides=2, padding='valid')

    #     #layer 2
    #     self.conv2 = Conv2D(256, 5, strides=1, padding='valid')
    #     self.bn2 = BatchNormalization(trainable=False)
    #     self.relu2 = ReLU()

    #     self.pool2 = MaxPool2D(pool_size=3, strides=2, padding='valid')

    #     #layer 3
    #     self.conv3 = Conv2D(384, 3, strides=1, padding='valid')
    #     self.bn3 = layers.BatchNormalization(trainable=False)
    #     self.relu3 = ReLU()

    #     #layer 4
    #     self.conv4 = Conv2D(384, 3, strides=1, padding='valid')
    #     self.bn4 = BatchNormalization(trainable=False)
    #     self.relu4 = ReLU()

    #     #layer 5
    #     self.conv5 = Conv2D(256, 3, strides=1, padding='valid')
    #     self.bn5 = BatchNormalization(trainable=False)

    # def forword(self, input):
    #     #layer 1
    #     x = input = Input([None, None, 3])
    #     x = self.conv1(input)
    #     x = self.bn1(x)
    #     x = self.relu1(x)

    #     x = self.pool1(x)

    #     #layer 2
    #     x = self.conv2(x)
    #     x = self.bn2(x)
    #     x = self.relu2(x)

    #     x = self.pool2(x)

    #     #layer 3
    #     x = self.conv3(x)
    #     x = self.bn3(x)
    #     x = self.relu3(x)

    #     #layer 4
    #     x = self.conv4(x)
    #     x = self.bn4(x)
    #     x = self.relu4(x)

    #     #layer 5
    #     x = self.conv5(x)
    #     output = self.bn5(x) 
    
    #     return output


            






















