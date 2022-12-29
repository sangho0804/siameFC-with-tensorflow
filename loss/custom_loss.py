from keras import backend as K

def loss_of_scoreMap(y_true, y_pred):
    product = -y_true * y_pred
    probs = 1 + K.clip(K.exp(product), 0, 1e6)
    loss = K.log(probs)
    return K.mean(loss, axis=(1,2))
