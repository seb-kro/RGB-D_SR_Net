# '''
# Created on 10.05.2016
# 
# @author: sebastian
# '''
# 
# import cv2
# import numpy as np
# from keras.layers import Input, Convolution2D, merge
# from keras.models import Model
# from keras.initializations import he_normal
# from keras.activations import relu
# from keras.layers.convolutional import UpSampling2D
# 
# 
# 
# #input tensor
# input_image = Input(shape=(4,None,None))
# 
# #contraction part
# conv1 = Convolution2D(64, 7, 7, dim_ordering='tf', init=he_normal, activation=relu, border_mode='same', subsample=(2, 2))(input_image)
# 
# conv2 = Convolution2D(128, 5, 5, dim_ordering='tf', init=he_normal, activation=relu, border_mode='same', subsample=(2, 2))(conv1)
# 
# 
# #extension part
# upconv2 = UpSampling2D(size=(2,2), dim_ordering='tf')(conv1)
# upconv2 = Convolution2D(128, 4, 4, dim_ordering='tf', init=he_normal, activation=relu, border_mode='same')(upconv2)
# upconv2 = Convolution2D(128, 3, 3, dim_ordering='tf', init=he_normal, activation=relu, border_mode='same')(upconv2)
# 
# upconv1 = merge([conv2, upconv2], mode='concat', concat_axis=-1)
# print('Does -1 correspond to last axis?')
# upconv1 = UpSampling2D(size=(2,2), dim_ordering='tf')(upconv1)
# upconv1 = Convolution2D(64, 4, 4, dim_ordering='tf', init=he_normal, activation=relu, border_mode='same')(upconv1)
# upconv1 = Convolution2D(64, 3, 3, dim_ordering='tf', init=he_normal, activation=relu, border_mode='same')(upconv1)
# 
# sr_prediction = merge([conv1, upconv1], mode='concat', concat_axis=-1)
# print('Does -1 correspond to last axis?')
# sr_prediction = UpSampling2D(size=(2,2), dim_ordering='tf')(sr_prediction)
# sr_prediction = Convolution2D(4, 4, 4, dim_ordering='tf', init=he_normal, activation=relu, border_mode='same')(sr_prediction)
# sr_prediction = Convolution2D(4, 3, 3, dim_ordering='tf', init=he_normal, activation=relu, border_mode='same')(sr_prediction)
# 
# 
# #model configuration
# model = Model(input=input_image, output=sr_prediction)
# model.compile(optimizer='rmsprop', loss='mse')
# 
# 
# #training
# img = cv2.imread()
# 
# x_train = np.random.random((100, 4, 500, 500))
# y_train = np.random.random((100, 4, 1000, 1000))
# 
# model.fit(x_train, y_train)


