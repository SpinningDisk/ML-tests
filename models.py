from keras import Sequential, layers, activations, Input

model01 = Sequential([
    Input(shape=(28, 28, 1)),
    layers.Conv2D(filters=32, kernel_size=(3, 3), kernel_initializer='random_normal',),
    layers.Activation('relu'),
    layers.MaxPool2D((2, 2), padding="same"),
    layers.Conv2D(filters=32, kernel_size=(3, 3), kernel_initializer='random_normal',),
    layers.Activation('relu'),
    layers.MaxPool2D((2, 2), padding="same"),
    layers.Conv2D(filters=32, kernel_size=(3, 3), kernel_initializer='random_normal',),
    layers.Activation('relu'),
    layers.MaxPool2D((2, 2), padding="same"),
    layers.Activation('relu'),
    layers.Flatten(),
    layers.Dense(32, kernel_initializer='random_normal',),
    layers.Dense(32, kernel_initializer='random_normal',),
    layers.Dense(16, kernel_initializer='random_normal',),
    layers.Dense(10, kernel_initializer='random_normal',),
    layers.Activation('softmax')
])