'''
Created on 20.05.2016

@author: sebastian
'''
import unittest
from misc.dataset import Dataset
from neuralnets import single_image_sr_net

'''
Module vars
'''
ds = None

'''
Module setup
'''
def setUpModule():
    global ds
    ds = Dataset()



class UntrainedTest(unittest.TestCase):

    def testUntrainedNet(self):
        global ds
        net = single_image_sr_net.SingleImageSrNet()
        (mse_keras, mse_scikit, predictions) = net.test(dataset=ds)
        assert mse_keras == mse_scikit
        print(mse_keras)
        pass



class TrainedTest(unittest.TestCase):

    def testTrainedNet(self):
        global ds
        print(ds.lr_inputs.shape)
        print(ds.sr_outputs.shape)
        net = single_image_sr_net.SingleImageSrNet()
        net.train(dataset=ds)
        (mse_keras, mse_scikit, predictions) = net.test(dataset=ds)
        assert mse_keras == mse_scikit
        print(mse_keras)
        pass
        pass

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()