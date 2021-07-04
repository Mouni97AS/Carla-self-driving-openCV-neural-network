# alexnet.py

'''
https://pythonprogramming.net/tflearn-machine-learning-tutorial/

TFLearn:
- abstraction layer on TensorFlow
- build: input layer -> conv1 -> conv2 -> fully connected l -> output l.
- what is alexnet: successfull network for image data

'''

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

def alexnet(width, height, lr):
    
    # Input layer                     (80  , 60)
    network = input_data(shape=[None, width, height, 1], name='input')

    # Convolution layer 1
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    # Convolution layer 2
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    # Fully connected layer 1
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)

    # Fully connected layer 2
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)

    # Output layer    
    network = fully_connected(network, 3, activation='softmax')
    # Calculating the loss: from input how close we are to the loss
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=lr, name='targets')

    model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                        max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='log')

    return model




""" AlexNet.
References:
    - Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton. ImageNet
    Classification with Deep Convolutional Neural Networks. NIPS, 2012.
Links:
    - [AlexNet Paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
"""
