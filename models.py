from keras import Sequential, layers, activations, Input, utils, datasets
from numpy import expand_dims

class num_rec_models():
    in_size = (28, 28, 1)
    out_size = 10
    def data_prep():
        num_classes = 10
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255
        x_train = expand_dims(x_train, -1)
        x_test = expand_dims(x_test, -1)
        y_train = utils.to_categorical(y_train, num_classes)
        y_test = utils.to_categorical(y_test, num_classes)
        
        return (x_train, y_train), (x_test, y_test)    
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
            layers.Activation('relu'),
            layers.Dense(32, kernel_initializer='random_normal',),
            layers.Activation('relu'),
            layers.Dropout(0.2),
            layers.Dense(16, kernel_initializer='random_normal',),
            layers.Activation('relu'),
            layers.Dense(10, kernel_initializer='random_normal',),
            layers.Activation('softmax')
        ])
        epochs = 4
        batch_size = 128
    class m02():
        model02 = Sequential([
            Input(shape=(28, 28, 1)),
            layers.Flatten(),
            layers.Dense(32, kernel_initializer='random_normal',),
            layers.Activation('relu'),
            layers.Dense(32, kernel_initializer='random_normal',),
            layers.Activation('relu'),
            layers.Dropout(0.125),
            layers.Dense(16, kernel_initializer='random_normal'),
            layers.Dense(10, kernel_initializer='random_normal',),
            layers.Activation('softmax'),
        ])
        epochs = 5
        batch_size = 128
#class img_rec_models():
    #def m01():
        #model01 = 