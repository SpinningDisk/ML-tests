from keras import Sequential, layers, activations, Input

model01 = Sequential([
    Input(shape=(28, 28)),
    layers.Conv2d(filters=32, kernel_size=(3, 3), padding="same"),
    layers.maxPool2D((2, 2)),
])