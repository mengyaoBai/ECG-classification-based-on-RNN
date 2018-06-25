import numpy as np
import keras
import keras.layers as layers
from keras.models import Model
from keras.models import Sequential
from keras.layers import SimpleRNN,LSTM,GRU
from keras.layers import Activation,Dense

from keras.optimizers import adam
from keras import regularizers

import random
import os
from os import listdir
from os.path import isfile, join
import gc


time_stemps=2
input_size=300
output_size=4
cell_size=60



def rnn_model(input_shape):
    

    model=Sequential()
    model.add(Dense(32,activation='relu',input_shape=(input_size,time_stemps)))
    model.add(LSTM(32,return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(output_size,activation='softmax'))



    #model.add(SimpleRNN(return_sequences=True,
    #                units=cell_size,
    #                input_dim=time_stemps,
    #                input_length=input_size))
    #model.add(SimpleRNN(return_sequences=True,
    #                units=cell_size))
    #model.add(SimpleRNN(units=cell_size))
    #model.add(Dense(output_size,activation='softmax'))



    adam_lr=0.005
    model.compile(optimizer = adam(lr=adam_lr) , loss='categorical_crossentropy', metrics=['accuracy'])
    return model

