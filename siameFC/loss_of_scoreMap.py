from keras import backend as K

# loss function
def loss_of_scoreMap(y_true, y_pred):
    product = -y_true * y_pred
    probs = 1 + K.clip(K.exp(product), 0, 1e6)
    loss = K.log(probs + 1e-7)
    return K.mean(loss)


