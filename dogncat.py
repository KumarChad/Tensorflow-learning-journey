from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
zip_dir = tf.keras.utils.get_file('cats_and_dogs_filtered.zip',origin=_URL, extract=True)

base_dir = os.path.join(os.path.dirname(zip_dir), 'cats_and_dogs_filtered')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

tcats = os.path.join(train_dir, 'cats')
tdogs = os.path.join(train_dir, 'dogs')
vcats = os.path.join(validation_dir, 'cats')
vdogs = os.path.join(validation_dir, 'dogs')

numtcats = len(os.listdir(tcats))
numtdogs = len(os.listdir(tdogs))

numvcats = len(os.listdir(vcats))
numvdogs = len(os.listdir(vdogs))

totalt = numtcats + numtdogs
totalv = numvcats + numvdogs

bsize = 100
isize = 150

imggen = ImageDataGenerator(rescale= 1/255, 
                            rotation_range=40, 
                            width_shift_range=0.2, 
                            height_shift_range=0.2, 
                            shear_range=0.2, 
                            zoom_range=0.2, 
                            horizontal_flip=True, 
                            fill_mode='nearest')

trainimg = ImageDataGenerator(rescale=1/255)
validimg = ImageDataGenerator(rescale=1/255)

traingen = imggen.flow_from_directory(batch_size=bsize, 
                                      directory=train_dir, 
                                      shuffle=True, 
                                      target_size=(isize, isize), 
                                      class_mode='binary')
validgen = validimg.flow_from_directory(batch_size=bsize, 
                                        directory=validation_dir, 
                                        shuffle=False, 
                                        target_size=(isize, isize), 
                                        class_mode='binary')


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D((2,2), strides=2),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2), strides=2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2), strides=2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2), strides=2),

    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()


epoch = 20
histroy = model.fit(
    traingen,
    steps_per_epoch = int(np.ceil(totalt / float(bsize))),
    epochs = epoch,
    validation_data = validgen,
    validation_steps = int(np.ceil(totalv / float(bsize)))
)


