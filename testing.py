import keras
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from PIL import Image
import warnings


num_classes = 10

model_name = input("what model?\n>")+'.keras'
try:
    model = keras.saving.load_model(f"models/{model_name}")
except:
    warnings.warn(f'no such model found "{model_name}"', stacklevel=3)
    exit(1)

req = input("random, user, eval (evaluation) or chosen test\n>>")
if req in ["random", "Random"]:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32") / 255
    x_test = np.expand_dims(x_test, -1)
    x_test = x_test.reshape((10000, 1, 28, 28, 1))
    #print(x_test.shape)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    i = randint(0, 10000)
    img = x_test[i]
    ans = y_test[i]
elif req in ["user", "User"]:
    path = input("path to image\n>>>")
    pre_img = Image.open(path).convert('L')
    numpydata = np.array(pre_img)
    try:
        img = np.expand_dims(numpydata, -1)
        img = numpydata.reshape(1, 28, 28, 1)
    except:
        warnings.warn(f"input image doesn not have the appropriate size to perform such action. \n(expected size of (28, 28), recieved {numpydata.shape})")
        exit(1)
    ans = input("what number is displayed?\n>>>>")
elif req in ["chosen test", "Chosen Test", "Chosen test", "chosen Test"]:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32") / 255
    x_test = np.expand_dims(x_test, -1)
    x_test = x_test.reshape((10000, 1, 28, 28, 1))
    #print(x_test.shape)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    i = int(input("number of test:\n>>>"))
    img = x_test[i]
    ans = y_test[i]
elif req in ['eval', 'Eval', 'e', 'E']:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32") / 255
    x_test = np.expand_dims(x_test, -1)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    model.evaluate(x_test, y_test)
    exit(0)

conf = model.predict(img).tolist()
try:
    ans = ans.tolist()
except:
    ans = keras.utils.to_categorical(ans, 10).tolist()
if conf[0].index(max(conf[0])) == ans.index(max(ans)):
    print(f"network guessed right ({conf[0].index(max(conf[0]))} = {ans.index(max(ans))})")
else:
    print(f"network did not guess right :( ({conf[0].index(max(conf[0]))} is not {ans.index(max(ans))})")

if input("show image?\n>>") in ["yes", "Yes", "y", "Y"]:
    img = img.reshape((28, 28))
    image = Image.fromarray(img)
    image.resize((28, 28))
    image.save("no.tiff")