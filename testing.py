import keras
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from PIL import Image
import warnings
import models


model_class = input("what class?\n>")
num_classes = eval(f'models.{model_class}.out_size')
model_num = input("what number?\n>")
size_in = eval(f'models.{model_class}.in_size')
try:
    model = keras.saving.load_model(f"models/{model_class}-model{model_num}.keras")
except:
    warnings.warn(f'no such model found "{model_class}-model{model_num}"', stacklevel=3)
    exit(1)

req = input("random, user, eval (evaluation) or chosen test\n>>")
if req in ["random", "Random"]:
    data = eval(f'models.{model_class}.data_prep()')
    (x_train, y_train), (x_test, y_test) = data
    i = randint(0, len(x_test))
    print(x_test.shape)
    img = x_test[i]
    ans = y_test[i]
    print(img.shape)
    print(ans.shape)
elif req in ["user", "User"]:
    path = input("path to image\n>>>")
    pre_img = Image.open(path).convert(eval(f"models.{model_class}.uom"))
    numpydata = np.array(pre_img)
    try:
        img = numpydata.reshape(eval(f"models.{model_class}.in_size"))
        img = np.expand_dims(numpydata, 0)
        print(img.shape)
    except:
        warnings.warn(f"input image doesn not have the appropriate size to perform such action. \n(expected size of (28, 28), recieved {numpydata.shape})")
        exit(1)
    ans = input("what number is displayed?\n>>>>")
elif req in ["chosen test", "Chosen Test", "Chosen test", "chosen Test"]:
    data = eval(f'models.{model_class}.data_prep()')
    (x_train, y_train), (x_test, y_test) = data
    i = int(input("number of test:\n>>>"))
    img = x_test[i]
    ans = y_test[i]
elif req in ['eval', 'Eval', 'e', 'E']:
    data = eval(f'models.{model_class}.data_prep()')
    (x_train, y_train), (x_test, y_test) = data
    model.evaluate(x_test, y_test)
    exit(0)
else:
    exit(1)

conf = model.predict(img).tolist()
try:
    ans = ans.tolist()
except:
    ans = keras.utils.to_categorical(ans, num_classes).tolist()
if conf[0].index(max(conf[0])) == ans.index(max(ans)):
    print(f"network guessed right ({eval(f'models.{model_class}.classes[conf[0].index(max(conf[0]))]')} = {eval(f'models.{model_class}.classes[ans.index(max(ans))]')})")
else:
    print(f"network did not guess right :( ({conf[0].index(max(conf[0]))} is not {ans.index(max(ans))})")

if input("show image?\n>>") in ["yes", "Yes", "y", "Y"]:
    #img = img.reshape(size_in)
    #image = Image.fromarray(img)
    #image.resize(size_in)
    #image.save("no.tiff")
    img = img.reshape(size_in)
    plt.imshow(img)
    plt.savefig("result.png")


if input("again?\n>>") in ["Y", "y"]:
    print("\n\n\n\n\n")
    exec(open("testing.py", "r").read())