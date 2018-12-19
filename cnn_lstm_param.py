# -*- coding: utf-8 -*-
#!/usr/bin/env python

from __future__ import print_function

import input_data
import keras.backend as K
from hyperopt import Trials, STATUS_OK, tpe
from keras.models import Sequential
from hyperas import optim
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import SGD , Adam, RMSprop, Adagrad
from keras.callbacks import EarlyStopping
from hyperas.distributions import choice, uniform, conditional


def data():
    """
    Data providing function:

    This function is separated from model() so that hyperopt
    won't reload data for each evaluation run.
    """
    train, validation, test = input_data.read_data_sets()

    flow_train = train.flow
    flow_train = flow_train.reshape((flow_train.shape[0], 8, 33, 1))
    labels_train = train.labels
    
    flow_validation = validation.flow
    flow_validation = flow_validation.reshape((flow_validation.shape[0], 8, 33, 1))
    labels_validation = validation.labels

    flow_test = test.flow
    flow_test = flow_test.reshape((flow_test.shape[0], 8, 33, 1))
    labels_test = test.labels
    return flow_train, labels_train, flow_test, labels_test

def model(flow_train, labels_train, flow_validation, labels_validation, flow_test, labels_test):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred-y_true), axis=-1))


    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=40,
                                     kernel_size=3,
                                     strides=1,
                                     padding='valid'), input_shape=[8, 33, 1]))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))

    model.add(TimeDistributed(Conv1D(filters=40,
                                     kernel_size=3,
                                     strides=1,
                                     padding='valid')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Conv1D(filters=40,
                                     kernel_size=2,
                                     strides=1,
                                     padding='valid')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Flatten()))	
    
    model.add(LSTM(units=64, return_sequences=True))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(LSTM(units=256)) 
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(33))
    model.compile(loss='mse', metrics=['mae', rmse, 'cosine'],
                  optimizer='rmsprop')
    callbacks = [EarlyStopping(patience=2)]

    model.fit(flow_train, labels_train,
              batch_size={{choice([8, 16, 32, 64])}},
              epochs=20,
              verbose=2,
              callbacks=callbacks,
              validation_data=(flow_validation, labels_validation))
    score = model.evaluate(flow_test, labels_test, verbose=0)
    acc = score[1]
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=model,
                                              data=data,
                                              algo=tpe.suggest,
                                              max_evals=2,
                                              trials=Trials())
    flow_train, labels_train, flow_test, labels_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(flow_test, labels_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)

    model_json = best_model.to_json()
    with open("cnn_lstm.json", "w") as json_file:
        json_file.write(model_json)
    best_model.save_weights("cnn_lstm.h5")
    print("Saved model to disk")
