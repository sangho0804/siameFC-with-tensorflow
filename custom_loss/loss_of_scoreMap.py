from keras import backend as K

def loss_of_scoreMap(y_true, y_pred):
    product = -y_true * y_pred
    probs = 1 + K.exp(product)
    loss = K.log(probs)
    return K.mean(loss)
