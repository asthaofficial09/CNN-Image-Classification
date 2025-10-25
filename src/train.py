
from tensorflow.keras import datasets
from src.model import build_cnn

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = build_cnn()
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
model.save("models/cnn_cifar10_model.h5")
