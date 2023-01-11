import tensorflow as tf

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops

#logistic loss
def logistic_fn(labels=None,logits=None):

    #convert tensor
    logits = ops.convert_to_tensor(logits, name="logits")
    labels = ops.convert_to_tensor(labels, name="labels")

    #zero tensor
    #for compared label value
    zeros = array_ops.zeros_like(logits, dtype=logits.dtype)
    
    #If y_true value(+1) >= 0 --> true, 
    #Else value is  (-1) then false.
    cond = (logits >= zeros)

    #used log(-yx) frame 
    # If cond (true) then -x, 
    # Else cond(False) then x 
    neg_abs_logits = array_ops.where(cond, -logits, logits)

    return math_ops.log1p(math_ops.exp(neg_abs_logits))


#total loss function
def loss_fn(y_true, y_pred):
    
    #use logistic_fn
    logistic = logistic_fn(labels=y_true, logits=y_pred)

    #mean
    loss = tf.reduce_mean(logistic, axis=[1,2])
    
    return loss



# # loss function
# # use sigmoid cross entropy with logits
# def loss_of_scoreMap(y_true, y_pred):

#     y_pred = tf.convert_to_tensor(y_pred)
#     y_true - tf.cast(y_true, y_pred.dtype)
#     y_true = tf.convert_to_tensor(y_true)
    
#     #logistic loss tensor
#     loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_true)
    
#     #mean
#     loss = tf.reduce_mean(loss, axis=[1,2])

#     return loss
