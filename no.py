import keras
import numpy as np

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_test = x_test.astype("float32") / 255
x_test = np.expand_dims(x_test, -1)

model = keras.saving.load_model("models/model01.keras")
layer_name = model.layers[4].name
intermediate_layer_model = keras.models.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
model.predict(x_test)
print(intermediate_layer_model)