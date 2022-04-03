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

"""Definition Function for preparing DATA:-"""

def dataProcess(batch_size, data_aug=True):
    if(data_aug):
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=90,
            shear_range=0.2,
            zoom_range=0.2,
            validation_split=0.1,
            horizontal_flip=True)
    else:
        train_datagen = ImageDataGenerator( rescale=1./255, validation_split=0.1)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        'inaturalist_12K/train',
        subset='training',
        target_size=(227,227),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle = True,
        seed = 45)

    val_generator = train_datagen.flow_from_directory(
        'inaturalist_12K/train',
        subset = 'validation',
        target_size=(227,227),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle = True,
        seed = 45)

    test_generator = test_datagen.flow_from_directory(
        'inaturalist_12K/val',
        target_size=(227,227),
        batch_size=BATCH_SIZE,
        class_mode='categorical')
    
    return train_generator, val_generator, test_generator



from sys import argv

if __name__ == "__main__":

    if(len(argv) != 15):
        print("Invalid num of parameters passed ")
        exit()

    fil=[0,0,0,0,0]
    for i in range(5):
        fil[i] = int(argv[i+1])

    fils = int(argv[6])

    BATCH_SIZE = int(argv[7])

    d = float(argv[8])

    if argv[9] == "True":
        batch_norm = True
    else:
        batch_norm = False

    epoch = int(argv[10])

    d_n = int(argv[11])
    ler = float(argv[12])
    act = argv[13]
    padd = argv[14]



    trial = create_cnn_model(fil,fils,d_n,batch_norm,d,act,padd,ler)

    train_gen, val_gen, test_gen = dataProcess(BATCH_SIZE, True)        #Data augmentation is True

    trial.fit(
              train_gen,
              steps_per_epoch = train_gen.samples // BATCH_SIZE,
              validation_data = val_gen, 
              validation_steps = val_gen.samples // BATCH_SIZE,
              epochs = epoch
            )

    trial.evaluate(test_gen,batch_size = BATCH_SIZE)

    """
    README Part-A - Q1 --------------------------------------------------------------------
        
        To compile the file with command line arguments write in following format in terminal :-

        $ python filename.py nf1 nf2 nf3 nf4 nf5 filter_size batch_size dropout batch_normalisation(True/False) epochs dense_neurons learning_rate activation padding
        
          nfi = number of filters in ith Layer
        =>nf3 = number of filters in 3rd layer

        
        Example:-
        $ python A2_PartA_Q1.py 8 16 32 64 128 5 32 0.15 True 2 32 0.001 relu same

        number of fileters in each convolution layer is flexible and can be given by user in command line.
    
    """
