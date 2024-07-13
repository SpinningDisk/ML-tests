import keras

model = keras.saving.load_model("models/model01.keras")
model.summary()