#Import the libraries

import tensorflow as tf
from tensorflow import keras
import numpy as np
import math
import matplotlib.pyplot as plt
import sys

from keras.layers import Dense, Flatten
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator


#Import the data

train_path = 'inaturalist_12K/train'
val_path = 'inaturalist_12K/val'


#Preprocess the data

def dataProcess(batch_size = 32, data_aug = True, image_size = [299,299]):

    # data augmentation
    if(data_aug):
        train_datagen = ImageDataGenerator(
            rescale = 1./255,
            rotation_range = 90,
            shear_range = 0.2,
            zoom_range = 0.2,
            validation_split = 0.1,
            horizontal_flip = True)
    else:
        train_datagen = ImageDataGenerator( rescale = 1./255, validation_split = 0.1)

    test_datagen = ImageDataGenerator(rescale = 1./255)

    # generate train dataset
    train_generator = train_datagen.flow_from_directory(
        train_path,
        subset='training',
        target_size = image_size,
        batch_size = batch_size,
        class_mode = 'categorical',
        shuffle = True,
        seed = 45)

    # generate validation dataset
    val_generator = train_datagen.flow_from_directory(
        train_path,
        subset = 'validation',
        target_size = image_size,
        batch_size = batch_size,
        class_mode = 'categorical',
        shuffle = True,
        seed = 45)

    # generate test dataset
    test_generator = test_datagen.flow_from_directory(
        val_path,
        target_size = image_size,
        batch_size = batch_size,
        class_mode = 'categorical')

    # plot one image of each class
    class_names = ['Amphibia', 'Animalia', 'Arachnida', 'Aves', 'Fungi',
               'Insecta', 'Mammalia', 'Mollusca', 'Plantae', 'Reptilia']
    plt.figure(figsize=(10, 10))
    images, labels = val_generator.next()
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(class_names[np.where(labels[i] == 1)[0][0]])
        plt.axis("off")
    plt.show()
    
    return train_generator, val_generator, test_generator


#Prepare the pre-trained model for our usage

def buildModel(pre_trained_model = 'InceptionV3',optimizer = 'adam', lr = 0.0001, image_size = [299,299], freeze = 1):
    
    # prepare the pre-trained model excluding the last layer
    if pre_trained_model == 'InceptionV3':
        pre_model = keras.applications.InceptionV3(input_shape = image_size + [3], weights = 'imagenet', include_top = False)
    if pre_trained_model == 'InceptionResNetV2':
        pre_model = keras.applications.InceptionResNetV2(input_shape = image_size + [3], weights = 'imagenet', include_top = False)
    if pre_trained_model == 'ResNet50':
        pre_model = keras.applications.ResNet50(input_shape = image_size + [3], weights = 'imagenet', include_top = False)
    if pre_trained_model == 'Xception':
        pre_model = keras.applications.Xception(input_shape = image_size + [3], weights = 'imagenet', include_top = False)

    # fine tuning the model
    k = len(pre_model.layers)
    upto = math.ceil(k*freeze)
    for i in range(upto):
        pre_model.layers[i] = False
    
    import os
    numberOfClasses = len(next(os.walk(train_path))[1])

    # flatten the last layer
    x = Flatten()(pre_model.output)
    # add output layer
    prediction = Dense(numberOfClasses, activation='softmax')(x)

    # create final model
    model = Model(inputs = pre_model.input, outputs = prediction)
    if optimizer == 'adam':
        opt = keras.optimizers.Adam(learning_rate = lr)
    if optimizer == 'adamax':
        opt = keras.optimizers.Adamax(learning_rate = lr)
    if optimizer == 'rmsprop':
        opt = keras.optimizers.RMSprop(learning_rate = lr)

    model.compile(optimizer = opt, 
                  loss = 'categorical_crossentropy', 
                  metrics = ['accuracy']
                  )
    return model

def train(epochs = 5, model = 'InceptionV3', optimizer = 'adamax', batch_size = 32,
            lr = 0.0001,data_aug = False, freeze = 1):

    # resize the images as per the pre trained model
    image_size = [299,299]
    if model == 'ResNet50' :
        image_size = [224,224]

    # process the data
    train_set, val_set, test_set = dataProcess(batch_size, data_aug, image_size)

    # build model 
    model = buildModel(model, optimizer, lr, image_size, freeze)

    # train the model and save the history
    trained_model = model.fit(train_set,
                              steps_per_epoch = train_set.samples // batch_size,
                              validation_data = val_set, 
                              validation_steps = val_set.samples // batch_size,
                              epochs = epochs
                              )

    # evaluate the model
    model.evaluate(test_set,
                   batch_size = batch_size
                  )

    return model, trained_model


if sys.argv[6] == 'True':
  daug = True
else :
  daug = False

#train(epochs = 5, model = 'InceptionV3', optimizer = 'adamax', batch_size = 32,lr = 0.0001,data_aug = False, freeze = 1)
# 5 InceptionV3 adamax 32 0.0001 False 0.9
train(int(sys.argv[1]),sys.argv[2],sys.argv[3],int(sys.argv[4]),float(sys.argv[5]),daug,float(sys.argv[7]))
