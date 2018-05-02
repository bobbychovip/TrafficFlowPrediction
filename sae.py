# -*- coding: utf-8 -*-
#!/usr/bin/env python

from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
import keras.backend as K
import input_data


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred-y_true), axis=-1))

time_steps = 8
batch_size = 64
hidden_layer1 = 256
hidden_layer2 = 256
hidden_layer3 = 33

train, validation, test = input_data.read_data_sets()

x_train = train.flow
x_train = x_train.reshape((-1, time_steps*33))
y_train = train.labels
x_test = test.flow
x_test = x_test.reshape((-1, time_steps*33))
y_test = test.labels
x_validation = validation.flow
x_validation = x_validation.reshape((-1, time_steps*33))
y_validation = validation.labels

input_img = Input(shape=(time_steps*33,))
encoded = Dense(hidden_layer1, activation='relu')(input_img)
encoded = Dense(hidden_layer2, activation='relu')(encoded)
encoded = Dense(hidden_layer3, activation='sigmoid')(encoded)
decoded = Dense(hidden_layer2, activation='relu')(encoded)
decoded = Dense(hidden_layer1, activation='relu')(decoded)
decoded = Dense(time_steps*33, activation='sigmoid')(decoded)
model = Model(input=input_img, output=encoded)

model.load_weights("Model/sae.h5")

model.compile(loss='mse', optimizer='rmsprop', metrics=['mae', rmse, 'cosine'])

"""
filepath = "Model/sae.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(x_train, y_train,
                epochs=10,
                batch_size=batch_size,
#                callbacks=callbacks_list,
                validation_data=(x_validation, y_validation))

"""
score = model.evaluate(x_test, y_test, verbose=1)
print('Test score:', score)

"""
model_json = model.to_json()
with open("Model/sae.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("Model/sae.h5")
print("Save model to disk")

json_file = open('Model/sae.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
sae_model = model_from_json(loaded_model_json)

sae_model.load_weights("Model/sae.h5")
"""
