import random, copy, time
import numpy as np
import cv2
from matplotlib import pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Conv2D, Dropout, 
                                     Flatten, MaxPooling2D)


class ModelBuilder():

    def __init__(self):

        self.models = {}
        self.modelParams = {}
        self.modelSummary = {}
        self.history = {}

        self.k = 0
        self.inputShape = None

    def setInputShape(self, input_shape):
        self.inputShape = input_shape

    def loadModels(self, models, modelParams, modelSummary):
        self.models = models
    
    def dumpModels(self):
        return (self.models, self.modelParams, self.modelSummary)

    def buildModel(self,
                   input_shape=None,
                   filters=28,
                   kernel_n=3,
                   dense_units=128,
                   dropout_frac=0.2,
                   lr=0.001
                   ):

        if input_shape is None:
            input_shape = self.inputShape

        if input_shape is None:
            print('need an input_shape / self.setInputShape()')
            return

        _model = Sequential()
        _model.add(Conv2D(filters, kernel_size=(kernel_n,kernel_n), 
                            input_shape=input_shape))
        _model.add(MaxPooling2D(pool_size=(2, 2)))
        _model.add(Flatten()) 
        _model.add(Dense(dense_units, activation=tf.nn.relu))
        _model.add(Dropout(dropout_frac))
        _model.add(Dense(1))
        
        opt = tf.keras.optimizers.RMSprop(lr)
        
        _model.compile(optimizer= opt, 
                    loss='mean_squared_error', 
                    metrics=[ 'mean_absolute_error'
                             ,'mean_squared_error'
                            ]
                    )

        _params = None
        
        k = str(self.k)
        self.k += 1

        self.models[k] = _model
        self.modelParams[k] = _params
        # self.modelSummary[k] = _model.summary()  #suppress print

    def fitModel(self, x, y, epochs=1, validation_split=0.0, k=None):
        
        if k is None:
            k = str(self.k - 1)

        _model = self.models[k]

        _history = _model.fit(x=x, y=y, epochs=epochs,
                              validation_split = validation_split
                              )

        self.history[str(self.k)] = _history

    @staticmethod
    def normData(img_list):
        
        data = img_list.copy()
        
        if len(data[0].shape) == 2:
            
            #only if it's grayscale
            len_data = data.shape[0]
            img_x, img_y = data[0].shape[0], data[0].shape[1]
        
            data = data.reshape(len_data, img_x, img_y, 1)
        
        data = data.astype('float32')
        
        return data

    @staticmethod
    def eval(model, x, y, names=False):
        ''' evaluate the model, use names=True to understand
            what each number in output list represents '''
        
        if names:
            names_list = []
            names_list.append(model.loss)
            names_list.extend(
                [_metric.name for _metric in model.metrics]
            )
            print('var_names for eval: ', str(names_list))
        
        eval_list = model.evaluate(x=x,y=y,verbose=0,)
        return eval_list

    @staticmethod
    def plotPredVsActual(truth, predicted):
        plt.scatter(list(truth), predicted)
        plt.xlabel('truth')
        plt.ylabel('predicted')
        high = plt.xlim()[1]
        plt.xlim([0, high])
        plt.ylim([0, high])
        plt.plot([-100, 100], [-100, 100])

    @staticmethod
    def plotHistory(history, y_high=5):

        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error')
        plt.plot(hist['epoch'], hist['mean_absolute_error'],
                label='Train Error')
        
        if hist.get('val_mean_absolute_error', None) is not None:
            plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
                    label = 'Val Error')
        
        plt.ylim([0,y_high])
        plt.legend()
        
        # plt.figure()
        # plt.xlabel('Epoch')
        # plt.ylabel('Mean Square Error [$MPG^2$]')
        # plt.plot(hist['epoch'], hist['mean_squared_error'],
        #         label='Train Error')
        # plt.plot(hist['epoch'], hist['val_mean_squared_error'],
        #         label = 'Val Error')
        # plt.ylim([0,20])
        # plt.legend()
        plt.show()
        
