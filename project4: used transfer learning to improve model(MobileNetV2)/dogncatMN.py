import time
import numpy as np
import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
tfds.disable_progress_bar

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import layers

from tensorflow.keras.preprocessing import image_dataset_from_directory

#test 1: keras layer was not working with sequential api
# dataset = image_dataset_from_directory(
#     'C:\\Users\\Admin\\Downloads\\cats_vs_dogs\\PetImages',
#     image_size=(224, 224),
#     batch_size=32,
#     label_mode='binary'
# )

# dataset = dataset.apply(tf.data.experimental.ignore_errors())

# def format(image, lbl):
#     image = tf.cast(image,tf.float32)/255.0
#     return image, lbl

# dataset = dataset.map(format)
# num_examples = tf.data.experimental.cardinality(dataset).numpy()
# bsize=32
# imgres=224

# train_size = int(0.8 * num_examples)
# val_size = num_examples - train_size

# traineg = dataset.take(train_size)
# valideg = dataset.skip(train_size)

# trainbatch = traineg.cache().shuffle(buffer_size=1000).batch(32).prefetch(tf.data.AUTOTUNE)
# validbatch = valideg.cache().batch(32).prefetch(tf.data.AUTOTUNE)

#test 1 for splitting data
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

bsize = 32
isize = 224

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


#test 2
# URL = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'
# featureext = hub.KerasLayer(URL, input_shape=(imgres,imgres,3))
# featureext.trainable =False

# model = tf.keras.Sequential([
#     featureext,
#     layers.Dense(2, activation='sigmoid')
# ])

# feature_ext = hub.load('https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4')

# def extract_feature(imagebatch):
#     return feature_ext(imagebatch)

# class custom_model(tf.keras.Model):
#     def __init__(self, feature_ext):
#         super(custom_model, self).__init__()
#         self.feature_ext = feature_ext
#         self.dense = tf.keras.layers.Dense(1,activation='sigmoid')

#     def call(self,inputs):
#         features = self.feature_ext(inputs)
#         return self.dense(features)
    
# model = custom_model(feature_ext)

# model.build(input_shape=(None,224,224,3))

#test 3
base_model = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1,activation='sigmoid')
])

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

epochs = 2
histroy = model.fit(traingen, epochs=epochs, validation_data=validgen)

t = time.time()
# export_path_keras = './{}.h5'.format(int (t))
# print(export_path_keras)

# model.save(export_path_keras)

