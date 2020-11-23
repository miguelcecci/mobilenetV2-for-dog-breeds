#!/bin/bash/python3.8

import tensorflow as tf
import numpy as np
import pickle
from glob import glob

classes = 120
loss = 'categorical_crossentropy'
activation = 'softmax'
epochs = 100

train_files = '/home/notoboto/Downloads/destino/train/'
test_files = '/home/notoboto/Downloads/destino/test/'
model_save_location = '/home/notoboto/Desktop/dogmobilenet/model'

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# creating scheduler
def scheduler(epoch, lr):
    if epoch < 60:
        return lr
    else:
        return 0.0001

callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

#loaded_model = './model'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
    )

test_datagen = ImageDataGenerator(
    rescale=1./255
    )


train_generator = train_datagen.flow_from_directory(
    train_files,
    target_size=(224, 224),
    batch_size = 20,
    class_mode = 'categorical'
    )

test_generator = test_datagen.flow_from_directory(
    test_files,
    target_size=(224, 224),
    batch_size = 20,
    class_mode = 'categorical'
    )


with open('./class_indices', 'wb') as file_pi:
    pickle.dump(train_generator.class_indices, file_pi)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model = tf.keras.applications.MobileNetV2(weights=None, classes = classes, classifier_activation=activation)

#overwriting model with loaded model
#if loaded_model != None:
#    model = tf.keras.models.load_model(loaded_model)

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

history = model.fit(train_generator, epochs=epochs, validation_data=test_generator, callbacks=[callback])

model.save(model_save_location)

with open('./trainHistoryDict', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

