import keras
import numpy as np
from keras import Sequential, layers, Input, initializers, optimizers
import visualkeras

num_classes = 10
input_shape = (28, 28, 1)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(x_train.shape)
print(y_train.shape)

model = Sequential([
    Input(shape=(28, 28, 1)),
    layers.Convolution2D(filters=2, kernel_size=(3, 3), activation="tanh"),
    layers.Dense(input_shape=(26, 26), units=676),
    layers.Convolution2D(filters=2, kernel_size=(3, 3), activation="tanh"),
    layers.Convolution2D(filters=2, kernel_size=(3, 3), activation="tanh"),
    keras.layers.Flatten(),
    layers.Dense(32, ),
    layers.Dense(16,),
    layers.Dense(10,)
])
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
#model.fit(x_train, y_train, batch_size=128, epochs=20)

visualkeras.layered_view(model, to_file='model01.png')

print("finished")