# Importing libraries

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D,Dense,Activation,MaxPooling2D,Flatten,BatchNormalization,Dropout,InputLayer
from tensorflow.keras.preprocessing.image import ImageDataGenerator



"""Function for creating CNN Model using parameters given"""
 
def create_cnn_model(filters, fil_size, dense_neuron, batch_norm, dropout, act, padd, lr):
    keras.backend.clear_session()
    model = Sequential()
    model.add(InputLayer(input_shape=(227,227,3))) #Adding Input layer with image size as(227,227)

    #Adding 5 convolution Layers in a loop
    for i in range(5):
        model.add(Conv2D(filters[i],(fil_size,fil_size),padding=padd))
        if batch_norm:
            model.add(BatchNormalization())   
        model.add(Activation(act))
        model.add(MaxPooling2D(pool_size=(2,2)))
    
    #Adding the Dense layers as per the parameters passed
    model.add(Flatten())
    model.add(Dense(dense_neuron))
    model.add(Dropout(dropout))
    model.add(Activation(act))

    model.add(Dense(10,activation='softmax'))
    #adam = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    return model
