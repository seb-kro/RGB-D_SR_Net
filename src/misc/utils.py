'''
Created on 19.05.2016

@author: sebastian
'''
import cv2
 
def show_image(image, window_name='image'):
    cv2.imshow(window_name, image / 255.0)

def test(message):
    '''
    Function for testing Sphinx documentation
        
    :param str message: The message to print. 
    '''
    print(message)