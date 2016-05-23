'''
Created on 21.05.2016

@author: sebastian
'''
import unittest
from misc.dataset import Dataset
from neuralnets.srcnn import Srcnn

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


@unittest.skip
class UntrainedTest(unittest.TestCase):

    def testTrainedNet(self):
        global ds
        net = Srcnn()
        (mse_keras, mse_scikit, predictions) = net.test(dataset=ds)
        assert mse_keras == mse_scikit
        print(mse_keras)
        pass



class TrainedTest(unittest.TestCase):

    def testUntrainedNet(self):
        global ds
        print(ds.lr_inputs.shape)
        print(ds.sr_outputs.shape)
        net = Srcnn()
        net.train(dataset=ds)
#         (mse_keras, mse_scikit, predictions) = net.test(dataset=ds)
#         assert mse_keras == mse_scikit
#         print(mse_keras)
#         pass
#         pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()