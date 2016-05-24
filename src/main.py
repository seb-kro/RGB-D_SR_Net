'''
Created on 18.05.2016

@author: sebastian
'''
from misc.dataset import Dataset
from misc import utils
import cv2

def main():
    '''
    Executes the application.
    '''
    print('Main');
    ds = Dataset()
    ds.read(ds.input_paths[0])
    lr_sample = ds.lr_inputs[0,:,:,:]
    sr_sample = ds.sr_outputs[0,:,:,:]
    print(ds.lr_inputs.shape)
    print(ds.sr_outputs.shape)
    print(lr_sample.shape)
    print(sr_sample.shape)
    lr_color_img = lr_sample[:,:,0:3]
    sr_color_img = sr_sample[:,:,0:3]
    lr_depth_img = lr_sample[:,:,3]
    sr_depth_img = sr_sample[:,:,3]
    # Scales float images to [0,1]
    utils.show_image(lr_color_img, 'lr color')
    utils.show_image(sr_color_img, 'sr color')
    utils.show_image(lr_depth_img, 'lr depth')
    utils.show_image(sr_depth_img, 'sr depth')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
if __name__ == '__main__':
    main()
