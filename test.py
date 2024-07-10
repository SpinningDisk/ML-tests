from keras import Sequential, layers, Input, initializers
from numpy import array as npar

calcs_ls = [
    [1, 3],
    [2, 4],
]

ans_ls = [
    [4],
    [6]
]

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