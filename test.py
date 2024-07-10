from keras import Sequential, layers, Input, initializers, optimizers
import visualkeras
from numpy import array as npar

rd = open("data.txt", "r").read().split("\n")
rs = open("ans.txt", "r").read().split("\n")
calcs_ls = []
ans_ls = []
counter = 0
for i in rd:
    calcs_ls.append(i.split("_"))
    for j in range(2):
        try:
            calcs_ls[counter][j-1] = int(calcs_ls[counter][j-1])
        except:
            pass
    try:
        ans_ls.append(int(rs[counter]))
    except:
        pass
    counter += 1
calcs_ls.pop(len(calcs_ls)-1)
print(calcs_ls)
print(ans_ls)

calcs = npar(calcs_ls)
ans = npar(ans_ls)

model = Sequential([
    Input(shape=(2,)),
    layers.Dense(1,),
])

model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])
model.fit(calcs, ans, batch_size=128, epochs=300);

for layer in model.layers:
    print(f"weights: {layer.weights}")

visualkeras.layered_view(model, to_file='model01.png')