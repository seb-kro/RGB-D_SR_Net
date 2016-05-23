'''
Created on 21.05.2016

@author: sebastian
'''

from .sr_net import SrNet
from keras.layers import Input, Convolution2D
from keras.models import Model
from keras.initializations import he_normal
from keras.activations import relu

class Srcnn(SrNet):
    '''
    SRCNN, a single image super resolution network suggested in 
    "Image Super-Resolution Using Deep Convolutional Networks" 
    by Chao Dong, Chen Change Loy, Kaiming He and Xiaoou Tang.
    
    As in the research article, the network is implemented with 
    filter sizes 9-3-1-5 and numbers of features 64-32-16.
    '''


    def __init__(self):
        '''
        Constructor
        '''
        super().__init__()
        # Defines the input tensor
        lr_image = Input(shape=(None,None,4))
        # Defines 4 convolutional layers
        conv = Convolution2D(64, 9, 9, dim_ordering='tf', init=he_normal, activation=relu, border_mode='same', subsample=(1, 1))(lr_image)       
        conv = Convolution2D(32, 3, 3, dim_ordering='tf', init=he_normal, activation=relu, border_mode='same', subsample=(1, 1))(conv)
        conv = Convolution2D(16, 1, 1, dim_ordering='tf', init=he_normal, activation=relu, border_mode='same', subsample=(1, 1))(conv)
        sr_prediction = Convolution2D(4, 5, 5, dim_ordering='tf', init=he_normal, activation=relu, border_mode='same', subsample=(1, 1))(conv)
        self.model = Model(input=lr_image, output=sr_prediction)
        self.model.compile(optimizer='rmsprop', loss='mse')
    