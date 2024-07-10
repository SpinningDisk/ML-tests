from keras import Sequential, layers, Input, initializers
from numpy import array as npar

rd = open("data.txt", "r").read().split("\n")
rs = open("ans.txt", "r").read().split("\n")
calcs_ls = []
ans_ls = []
counter = 0
for i in rd:
    calcs_ls.append(i.split("_"))
    for j in range(2):
        print(calcs_ls[counter][j-1])
        calcs_ls[counter][j-1] = int(calcs_ls[counter][j-1])
    ans_ls.append(int(rs[counter]))
    counter += 1
print(calcs_ls)
print(ans_ls)

exit()

calcs = npar(calcs_ls)
ans = npar(ans_ls)

model = Sequential([
    Input(shape=(3,)),
    layers.Dense(16, kernel_initializer='random_normal', bias_initializer='zeros'),
    layers.Activation('relu'),
    layers.Dense(16, kernel_initializer='random_normal', bias_initializer='zeros'),
    layers.Activation('relu'),
    layers.Dense(32, kernel_initializer='random_normal', bias_initializer='zeros'),
    layers.Activation('relu'),
    layers.Dense(10, kernel_initializer='random_normal', bias_initializer='zeros'),
    layers.Activation('tanh'),
    layers.Dense(1, kernel_initializer='random_normal', bias_initializer='zeros'),
])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(calcs, ans, batch_size=128, epochs=50);