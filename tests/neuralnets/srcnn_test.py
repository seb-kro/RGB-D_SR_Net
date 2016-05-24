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

    def testUntrainedNet(self):
        global ds
        net = Srcnn()
        (mse_keras, mse_scikit, predictions) = net.test(dataset=ds)
        assert mse_keras == mse_scikit
        print(mse_keras)
        pass



class TrainedTest(unittest.TestCase):
    
    @unittest.skip
    def testTrainedNet(self):
        global ds
        print(ds.lr_inputs.shape)
        print(ds.sr_outputs.shape)
        net = Srcnn()
        print('train')
        net.train(dataset=ds)
        print('test')
        (mse_keras, mse_scikit, predictions) = net.test(dataset=ds)
        print(mse_keras)
        assert mse_keras == mse_scikit
        pass
    
    def testTraining(self):
        global ds
#         print(ds.lr_inputs.shape)
#         print(ds.sr_outputs.shape)
        net = Srcnn()
        print('train')
        epoch = 1
        while True:
            print('Epoch ' + str(epoch) + ' ...')
            for input_path in ds.input_paths:
                ds.read(input_path)
                net.train(dataset=ds)
            net.save_weights('data/test/srcnn/weights/epoch' + str(epoch))
            epoch += 1
            
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()