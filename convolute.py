import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import PIL
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
#x_train.dtype = 'bfloat16'
#x_test.dtype = 'bfloat16'
x_train = tf.cast(x_train, dtype=tf.float32)
x_test = tf.cast(x_test, dtype=tf.float32)
x_train = np.expand_dims(x_train, 1)
x_test = np.expand_dims(x_test, 1)
#print(x_test.shape)
#print(x_train.shape)
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model = keras.saving.load_model("models/img_rec_models10-model01.keras")
#model.predict(x_test[1])
#layer_output = keras.Function(model.layers[0].input, model.layers[1].output)
#output = layer_output(x_test[1])
#print("finished")
#output = np.reshape(output, (30, 30, 32))
#megamap = np.mean(output, axis=2)
#print('got to image creation proccess')
#for i in range(32):
    #to_display = output[:, :, i]
    #plt.imshow(to_display, cmap='jet')
    #plt.savefig(f"conv_layers/feature{i}.png") 

#plt.imshow(megamap, cmap='jet')
#plt.savefig("conv_layers/megafeatures.png")

plt.imshow(x_test[1].reshape(32, 32, 3)/255)
plt.savefig("conv_layers/what_I_see.png")