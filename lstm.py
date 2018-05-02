# -*- coding: utf-8 -*-
#!/usr/bin/env python

from __future__ import print_function

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LSTM 
from keras.layers.wrappers import TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
import keras.backend as K
import input_data

BATCH_SIZE = 64
TIME_STEPS = 8
INPUT_SIZE = 33
OUTPUT_SIZE = 33

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred-y_true), axis=-1))

print('loading data...')

train, validation, test = input_data.read_data_sets()

flow_train = train.flow
labels_train = train.labels
flow_test = test.flow
labels_test = test.labels
flow_validation = validation.flow
labels_validation = validation.labels
print('build model...')

model = Sequential()
model.add(LSTM(input_shape=(TIME_STEPS, INPUT_SIZE),
               output_dim=64, 
               return_sequences=True,))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(LSTM(output_dim=256))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(OUTPUT_SIZE))

model.summary()

model.load_weights("Model/lstm.h5")

model.compile(loss='mse', optimizer='rmsprop', metrics=['mae', rmse, 'cosine'])

"""
filepath = "Model/lstm.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

print('Train...')

model.fit(flow_train, labels_train, 
          epochs=10, 
          callbacks=callbacks_list,
          batch_size= BATCH_SIZE, 
          validation_data=(flow_validation, labels_validation))

"""
score = model.evaluate(flow_test, labels_test, verbose=0)
print('Test socre:', score)

"""
model_json = model.to_json()
with open("Model/lstm.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("Model/lstm.h5")
print("Save model to disk")

json_file = open('Model/lstm.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
lstm_model = model_from_json(loaded_model_json)

lstm_model.load_weights("Model/lstm.h5")
"""
