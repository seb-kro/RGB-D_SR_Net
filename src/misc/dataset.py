'''
Created on 12.05.2016

@author: sebastian
'''
import cv2
import glob
import numpy as np

class Dataset(object):
    '''
    Set of training or testing images.
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.input_path = 'data/monkaa/frames_cleanpass_webp/eating_x2/left/'
        self.lr_inputs = None
        self.sr_outputs = None
        self.read();
    
    def read(self):
        '''
        Reads in the data from input files
        '''
        filenames = glob.glob(self.input_path + '*')
        #TODO: remove assertion
        assert len(filenames) > 0
        filenames.sort()
        outputs = []
        inputs = []
        for filename in filenames:
            output_img = cv2.imread(filename)
            # Asserts the image is read correctly and not empty
            assert output_img.shape[0] > 0
            assert output_img.shape[1] > 0
            #TODO: read in actual depth
            output_depth = np.random.random((output_img.shape[0], output_img.shape[1], 1))
            #print(type(output_img))
            output_img = np.concatenate((output_img, output_depth), 2)
            #print(type(output_img))
            outputs.append(output_img)
            input_img = compute_lr_input(
                                         output_img, downsampling_factor_x=2, 
                                         downsampling_factor_y=2, blur_sigma=1.6, noise_sigma=0.03)
            inputs.append(input_img)
        self.sr_outputs = np.stack(outputs, axis=0)
        self.lr_inputs = np.stack(inputs, axis=0)
                
        
        
def compute_lr_input(sr_image, downsampling_factor_x, downsampling_factor_y, blur_sigma, noise_sigma):
    '''
    Computes a synthetic lower resolution input image for a given super resolution ground truth
    '''
    # Blurs the image. GaussianBlur() allows images with arbitrary number of channels. Kernel size is derived from sigma.
    blurred_image = cv2.GaussianBlur(sr_image, ksize=(0,0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    # Downsamples the image
    lr_size = (int(round(blurred_image.shape[1]/downsampling_factor_x)), int(round(blurred_image.shape[0]/downsampling_factor_y)))
    downsampled_image = cv2.resize(blurred_image, lr_size, interpolation = cv2.INTER_AREA)
    # Asserts x dimension is scalable without remainder
    assert sr_image.shape[0] == downsampled_image.shape[0] * downsampling_factor_x
    # Asserts y dimension is scalable without remainder
    assert sr_image.shape[1] == downsampled_image.shape[1] * downsampling_factor_y
    # Adds Gaussian noise to the image
    noisy_image = downsampled_image + np.random.normal(0.0, noise_sigma, (downsampled_image.shape[0], downsampled_image.shape[1], 4))
    # Upsamples the image back to its original size
    upsampled_image = upsample_lr_input(noisy_image, downsampling_factor_x, downsampling_factor_y)
    return upsampled_image
    
def upsample_lr_input(lr_image, upsampling_factor_x, upsampling_factor_y):
    '''
    Upsamples an image via bicubic interpolation
    '''
    new_size = (upsampling_factor_x * lr_image.shape[1], upsampling_factor_y * lr_image.shape[0])
    upsampled_image = cv2.resize(lr_image, new_size, interpolation = cv2.INTER_CUBIC)
    return upsampled_image