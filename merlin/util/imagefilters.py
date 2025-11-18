import cv2
import numpy as np

"""
This module contains code for performing filtering operations on images
"""


def high_pass_filter(image: np.ndarray,
                     windowSize: int,
                     sigma: float) -> np.ndarray:
    """
    Args:
        image: the input image to be filtered
        windowSize: the size of the Gaussian kernel to use.
        sigma: the sigma of the Gaussian.

    Returns:
        the high pass filtered image. The returned image is the same type
        as the input image.
    """
    lowpass = cv2.GaussianBlur(image,
                               (windowSize, windowSize),
                               sigma,
                               borderType=cv2.BORDER_REPLICATE)
    gauss_highpass = image - lowpass
    gauss_highpass[lowpass > image] = 0
    return gauss_highpass

def Remove_Hot_Pixels(im, dtype=np.uint16, hot_pix_th=0.50, hot_th=4, 
                      interpolation_style='nearest', verbose=False):
    '''Function to remove hot pixels by interpolation in each single layer'''
    if verbose:
        print("-- removing hot pixels")
    # create convolution matrix, ignore boundaries for now
    _conv = (np.roll(im,1,1)+np.roll(im,-1,1)+np.roll(im,1,2)+np.roll(im,1,2))/4
    # hot pixels must be have signals higher than average of neighboring pixels by hot_th in more than hot_pix_th*total z-stacks
    _hotmat = im > hot_th * _conv
    _hotmat2D = np.sum(_hotmat,0)
    _hotpix_cand = np.where(_hotmat2D > hot_pix_th*np.shape(im)[0])
    # if no hot pixel detected, directly exit
    if len(_hotpix_cand[0]) == 0:
        return im
    # create new image to interpolate the hot pixels with average of neighboring pixels
    _nim = im.copy()
    if interpolation_style == 'nearest':
        for _x, _y in zip(_hotpix_cand[0],_hotpix_cand[1]):
            if _x > 0 and  _y > 0 and _x < im.shape[1]-1 and  _y < im.shape[2]-1:
                _nim[:,_x,_y] = (_nim[:,_x+1,_y]+_nim[:,_x-1,_y]+_nim[:,_x,_y+1]+_nim[:,_x,_y-1])/4
    return _nim.astype(dtype)