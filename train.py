import keras
import numpy as np
from keras import Sequential, layers, Input, initializers, optimizers, Initializer
import visualkeras
import json
import models

model_class = input("model class:\n>")
model_num = input("model numder:\n>>")
model = eval(f"models.{model_class}.m{model_num}.model{model_num}")
num_classes = eval(f'models.{model_class}.out_size')
input_shape = eval(f'models.{model_class}.in_size')

data = eval(f'models.{model_class}.data_prep()')
(x_train, y_train), (x_test, y_test) = data
print(x_train.shape)




model.compile(loss="categorical_crossentropy", optimizer="adamw", metrics=["accuracy"])
visualkeras.layered_view(model, to_file=f'model_pics/{model_class}-model{model_num}.png')
model.fit(x_train, y_train, batch_size=eval(f'models.{model_class}.m{model_num}.batch_size'), epochs=eval(f'models.{model_class}.m{model_num}.epochs'), validation_data=(x_test, y_test), validation_batch_size=5)
if input("overwrite?\n>>") in ["yes", "Yes", "y", "Y"]:
    model.save(f"models/{model_class}-model{model_num}.keras")

model.summary()
model.evaluate(x_test, y_test)