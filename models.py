from keras import Sequential, layers, activations, Input

class num_rec_models():
    class m01():
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
        epochs = 3
        batch_size = 128
    input_size = (28, 28, 1)
    out_size = 10
    
    def m02():
        model02 = Sequential([
            Input(shape=(28, 28, 1)),
            layers.MaxPool2D((2, 2),)
        ])
#class img_rec_models():
    #def m01():
        #model01 = 