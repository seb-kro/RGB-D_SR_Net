'''
Created on 21.05.2016

@author: sebastian
'''

from sklearn.metrics import mean_squared_error

class SrNet(object):
    '''
    Base class for super resolution neural networks.
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.model = None
        
    def train(self, dataset):
        '''
        Trains the network.
        
        :param Dataset dataset: Dataset of training data.
        '''
        self.model.fit(dataset.lr_inputs, dataset.sr_outputs, batch_size=32, nb_epoch=10, verbose=2)
        
    def test(self, dataset):
        '''
        Tests the network.
        
        :param Dataset dataset: Dataset of test data.
        '''
        mse_keras = self.model.evaluate(dataset.lr_inputs, batch_size=32, verbose=0)
        predictions = self.model.predict(dataset.lr_inputs, batch_size=32, verbose=0)
        mse_scikit = mean_squared_error(dataset.sr_outputs.flatten(), predictions.flatten())
        return (mse_keras, mse_scikit, predictions)