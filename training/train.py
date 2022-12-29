from loss.loss import loss_exp_fn
from siameFC.SiameFC import SiameseFC

model = make_model(X_SHAPE, Z_SHAPE)
model.compile(optimizer='adam', loss=loss_exp_fn, metrics=['accuracy'])

batch_size = 5
epochs = 2
model.fit([x_images, z_images], [labels], batch_size=batch_size, epochs=epochs)

