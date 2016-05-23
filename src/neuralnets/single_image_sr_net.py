'''
Created on 19.05.2016

@author: sebastian
'''

from keras.layers import Input, Convolution2D, merge
from keras.models import Model
from keras.initializations import he_normal
from keras.activations import relu
from keras.layers.convolutional import UpSampling2D
from .sr_net import SrNet

class SingleImageSrNet(SrNet):
    '''
    Encoder-decoder CNN for single image RGB-D super resolution.
    '''


    def __init__(self):
        '''
        Constructor
        '''
        super().__init__()
        # input tensor
        lr_image = Input(shape=(None,None,4))
        # contraction part
        conv1 = Convolution2D(64, 7, 7, dim_ordering='tf', init=he_normal, activation=relu, border_mode='same', subsample=(2, 2))(lr_image)       
        conv2 = Convolution2D(128, 5, 5, dim_ordering='tf', init=he_normal, activation=relu, border_mode='same', subsample=(2, 2))(conv1)          
        # extension part
        # upconvolution level 2
        upconv2 = UpSampling2D(size=(2,2), dim_ordering='tf')(conv1)
        upconv2 = Convolution2D(128, 4, 4, dim_ordering='tf', init=he_normal, activation=relu, border_mode='same')(upconv2)
        upconv2 = Convolution2D(128, 3, 3, dim_ordering='tf', init=he_normal, activation=relu, border_mode='same')(upconv2)
        # upconvolution level 1
        upconv1 = merge([conv2, upconv2], mode='concat', concat_axis=-1)
        #TODO: Does -1 correspond to last axis?
        print('Does -1 correspond to last axis?')
        upconv1 = UpSampling2D(size=(2,2), dim_ordering='tf')(upconv1)
        upconv1 = Convolution2D(64, 4, 4, dim_ordering='tf', init=he_normal, activation=relu, border_mode='same')(upconv1)
        upconv1 = Convolution2D(64, 3, 3, dim_ordering='tf', init=he_normal, activation=relu, border_mode='same')(upconv1)
        # upconvolution level 0 (sr)
        sr_prediction = merge([conv1, upconv1], mode='concat', concat_axis=-1)
        #TODO: Does -1 correspond to last axis?
        print('Does -1 correspond to last axis?')
        sr_prediction = UpSampling2D(size=(2,2), dim_ordering='tf')(sr_prediction)
        sr_prediction = Convolution2D(4, 4, 4, dim_ordering='tf', init=he_normal, activation=relu, border_mode='same')(sr_prediction)
        sr_prediction = Convolution2D(4, 3, 3, dim_ordering='tf', init=he_normal, activation=relu, border_mode='same')(sr_prediction)
        # configuration
        self.model = Model(input=lr_image, output=sr_prediction)
        self.model.compile(optimizer='rmsprop', loss='mse')