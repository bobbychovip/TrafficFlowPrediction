# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
把数据集分割为训练集、验证集和测试集
"""

import numpy as np
import data_preprocess
from tensorflow.contrib.learn.python.learn.datasets import base
from sklearn.cross_validation import train_test_split

class DataSet(object):
    def __init__(self, flow, labels):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._flow = flow
        self._labels = labels
        self._num_examples = flow.shape[0]
        pass

    @property
    def flow(self):
        return self._flow

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle = True):
        """
        Return the next 'batch_size' examples from this dataset
        """
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if start == 0 and self._epochs_completed == 0 and shuffle:
            idx = np.arange(self._num_examples)
            np.random.shuffle(idx)
            self._flow = self.flow[idx]
            self._labels = self.labels[idx]

        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            flow_rest_part = self._flow[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]

            # Shuffle the data
            if shuffle:
                idx0 = numpy.arrange(self._num_examples)
                numpy.random.shuffle(idx0)
                self._flow = self.flow[idx0]
                self._labels = self.labels[idx0]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            flow_new_part = self._flow[start:end]
            labels_new_part = self._labels[start:end]
            return numpy.concatenate((flow_rest_part, flow_new_part), axis=0), numpy.concatenate((labels_rest_part, labels_new_part), axis=0)

        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._flow[start:end], self._labels[start:end]

def create_data_sets():
    samples = data_preprocess.samples
    look_back = 8
    interval = 0
    flow, labels = [], []
    for i in range(len(samples)-look_back-interval):
        flow.append(samples[i:(i+look_back)])
        labels.append(samples[i+look_back+interval])
    return np.asarray(flow), np.asarray(labels)

def read_data_sets():
    flow, labels = create_data_sets()
    validation_size = 8000
    train_flow, test_flow, train_labels, test_labels = train_test_split(flow, 
                                                                        labels, 
                                                                        test_size = 0.2,
                                                                        random_state=0)
    validation_flow = train_flow[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_flow = train_flow[validation_size:]
    train_labels = train_labels[validation_size:]

    train = DataSet(train_flow, train_labels)
    test = DataSet(test_flow, test_labels)
    validation = DataSet(validation_flow, validation_labels)
    return base.Datasets(train=train, validation=validation, test=test)

train, validation, test = read_data_sets()
print(train.flow.shape)
