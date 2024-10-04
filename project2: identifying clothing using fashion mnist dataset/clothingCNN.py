from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow_datasets as tfds
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import math
import numpy as np
import matplotlib.pyplot as plt

import tqdm
import tqdm.auto
tqdm.tqdm = tqdm.auto.tqdm

dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
trainset, testset = dataset['train'], dataset['test']

train_eg = metadata.splits['train'].num_examples
test_eg = metadata.splits['test'].num_examples

def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels

trainset = trainset.map(normalize)
testset = testset.map(normalize)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu,input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2), strides=2),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2,2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

batchsize = 32
trainset = trainset.repeat().shuffle(train_eg).batch(batchsize)
testset=testset.batch(batchsize)

model.fit(trainset, epochs=10, steps_per_epoch=math.ceil(train_eg/batchsize))

test_loss, test_accuracy = model.evaluate(testset, steps=math.ceil(test_eg/32))
print('Accuracy on test dataset: ', test_accuracy)
