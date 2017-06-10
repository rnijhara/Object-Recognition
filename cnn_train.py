# -*- coding: utf-8 -*-

"""
Based on the tflearn example located here:
https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_cifar10.py
"""
from __future__ import division, print_function, absolute_import

# Import tflearn and some helpers
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
#from cnn_train_test import x_data, y_data, x_test_data, y_test_data
# import pickle
# import numpy
#
# def unpickle(file):
#     import cPickle
#     fo = open(file, 'rb')
#     dict = cPickle.load(fo)
#     fo.close()
#     return dict
#
# file1 = "/Users/roopaknijhara/Downloads/cifar/train"
#
# file2 = "/Users/roopaknijhara/Downloads/cifar/meta"
#
# pic_dict = unpickle(file1)
# meta_dict = unpickle(file2)
#
# print(pic_dict.keys())
#
# #print pic_dict['fine_labels']
#
# #print pic_dict['data']
#
# print(meta_dict.keys())
#
# print(meta_dict['coarse_label_names'])
#
# # 3 is food_containers, 9 is bottle
# y_data = list()
# i = 0
# for label in pic_dict['coarse_labels']:
#     if i == 500:
#         break
#     if label != 3:
#         y_data.append(numpy.reshape(pic_dict['data'][i].astype(float), (3, 32, 32)))
#     i += 1
#
# print(y_data)
# print(len(y_data))
#
# x_data = list()
# i = 0
# for label in pic_dict['fine_labels']:
#     if label == 9:
#         x_data.append(numpy.reshape(pic_dict['data'][i].astype(float), (3, 32, 32)))
#     i += 1
#
# print(x_data)
# print(len(x_data))
#
# file3 = "/Users/roopaknijhara/Downloads/cifar/test"
# test_dict = unpickle(file3)
#
#
# y_test_data = list()
# i = 0
# for label in test_dict['coarse_labels']:
#     if i == 100:
#         break
#     if label != 3:
#         y_test_data.append(numpy.reshape(test_dict['data'][i].astype(float), (3, 32, 32)))
#     i += 1
#
# print(y_test_data)
# print(len(y_test_data))
#
# x_test_data = list()
# i = 0
# for label in test_dict['fine_labels']:
#     if label == 9:
#         x_test_data.append(numpy.reshape(test_dict['data'][i].astype(float), (3, 32, 32)))
#     i += 1
#
# x_test_data = x_test_data[:96]
# print (x_test_data)
# print (len(x_test_data))


# Load the data set
#X, Y, X_test, Y_test = pickle.load(open("full_dataset.pkl", "rb"))

# Shuffle the data
# X, Y = shuffle(x_data[:481], y_data)

# Make sure the data is normalized

from tflearn.datasets import cifar100
(X, Y), (X_test, Y_test) = cifar100.load_data()

X, Y = shuffle(X, Y)
Y = to_categorical(Y, 2)
Y_test = to_categorical(Y_test, 2)

img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Create extra synthetic training data by flipping, rotating and blurring the
# images on our data set.
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)

# Define our network architecture:

# Input is a 32x32 image with 3 color channels (red, green and blue)
network = input_data(shape=[None, 32, 32, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

# Step 1: Convolution
network = conv_2d(network, 32, 3, activation='relu')

# Step 2: Max pooling
network = max_pool_2d(network, 2)

# Step 3: Convolution again
network = conv_2d(network, 64, 3, activation='relu')

# Step 4: Convolution yet again
network = conv_2d(network, 64, 3, activation='relu')

# Step 5: Max pooling again
network = max_pool_2d(network, 2)

# Step 6: Fully-connected 512 node neural network
network = fully_connected(network, 512, activation='relu')

# Step 7: Dropout - throw away some data randomly during training to prevent over-fitting
network = dropout(network, 0.5)

# Step 8: Fully-connected neural network with two outputs (0=isn't a bottle, 1=is a bottle) to make the final prediction
network = fully_connected(network, 2, activation='softmax')

# Tell tflearn how we want to train the network
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Wrap the network in a model object
model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='bottle-classifier.tfl.ckpt')

# Train it! We'll do 100 training passes and monitor it as it goes.
model.fit(X, Y, n_epoch=100, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=96,
          snapshot_epoch=True,
          run_id='bottle-classifier')

# Save model when training is complete to a file
model.save("bottle-classifier.tfl")
print("Network trained and saved as bottle-classifier.tfl!")
