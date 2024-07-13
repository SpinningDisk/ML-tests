import keras
import numpy as np
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_test = x_test.astype("float32") / 255
x_test = np.expand_dims(x_test, -1)
x_test = x_test.reshape((10000, 1, 28, 28, 1))

model = keras.saving.load_model("models/model01.keras")
model.predict(x_test[1])
layer_output = keras.Function(model.layers[0].input, model.layers[13].output)
output = layer_output(x_test[1])
output = np.reshape(output, (4, 4))
#megamap = np.mean(output, axis=2)
#for i in range(32):
#    to_display = output[:, :, i]
#    plt.imshow(to_display, cmap='jet')
#    plt.savefig(f"conv_layers/feature{i}.png") 

#plt.imshow(megamap, cmap='jet')
#plt.savefig("conv_layers/megafeatures.png")

plt.imshow(output, cmap='grey')
plt.savefig("conv_layers/neuronlayers.png")