import keras
from keras import utils
from numpy import expand_dims

model = keras.saving.load_model("models/num_rec_models-model01.keras")

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

num_classes = 10
#x_train = x_train.astype("float32") / 255
#x_test = x_test.astype("float32") / 255
print(x_train.shape)
x_train = expand_dims(x_train, 1)
x_test = expand_dims(x_test, 1)
print(x_train.shape)
#x_train = np.zeros((60000, 1, 1024, 768, 1))
#x_test = np.zers((10000, 1, 1024, 768, 1))
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

model.predict(x_test[1])