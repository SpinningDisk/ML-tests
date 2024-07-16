from keras import Sequential, layers, activations, Input, utils, datasets
from numpy import expand_dims
import numpy as np
 
class num_rec_models():
    in_size = (28, 28, 1)
    out_size = 10
    classes = {
        0: "0",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        8: "8",
        9: "9"
    }
    uom = 'L'
    def data_prep():
        num_classes = 10
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255
        x_train = expand_dims(x_train, -1)
        x_test = expand_dims(x_test, 1)
        #x_train = np.zeros((60000, 1, 1024, 768, 1))
        #x_test = np.zers((10000, 1, 1024, 768, 1))
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
class img_rec_models10():
    out_size = 10
    in_size  = (32, 32, 3)
    classes = {
        0: "airplane",
        1: "car",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck",
    }
    uom = 'RGB'
    def data_prep():
        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
        x_test = x_test.reshape((10000, 1, 32, 32, 3))
        y_train = utils.to_categorical(y_train, 10)
        y_test = utils.to_categorical(y_test, 10)
        return (x_train, y_train), (x_test, y_test)
    class m01():
        model01 = Sequential([
            Input(shape=(32, 32, 3)),
            layers.SeparableConv2D(filters=32, kernel_size=(3, 3),),
            layers.Activation('relu'),
            layers.MaxPool2D((2, 2)),
            layers.Dropout(0.2),
            layers.SeparableConv2D(filters=32, kernel_size=(3, 3),),
            layers.Activation('relu'),
            layers.MaxPool2D((2, 2)),
            layers.SeparableConv2D(filters=16, kernel_size=(2, 2),),
            layers.Activation('relu'),
            #layers.Reshape((20, 20, 1)),
            #layers.Conv2D(32, (5, 5), kernel_initializer='random_normal'),
            #layers.Activation('relu'),
            #layers.MaxPool2D((2, 2)),
            #layers.Conv2D(16, (2, 2), kernel_initializer='random_normal'),
            #layers.Activation('relu'),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(5, kernel_initializer='random_normal'),
            layers.Activation('relu'),
            layers.Dense(16, kernel_initializer='random_normal'),
            layers.Activation('relu'),
            layers.Dense(10, kernel_initializer='random_normal'),
            layers.Activation('softmax')
        ])
        epochs = 6
        batch_size = 1