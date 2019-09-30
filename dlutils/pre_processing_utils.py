import numpy as np
from scipy.ndimage.filters import gaussian_filter
import random

def sometimes(func, prob=0.5):
    return lambda im:func(im) if random.random()<prob else im

def apply_successively(*functions):
    def func(img):
        for f in functions:
            img = f(img)
        return img
    return func

def gaussian_blur(img, sig):
    return np.expand_dims(gaussian_filter(img.squeeze(-1), sig), -1)

def random_gaussian_blur(img, sig_min=1, sig_max=2):
    sig = random.uniform(sig_min, sig_max)
    return gaussian_blur(img, sig)

def adjust_histogram_range(img, min=0, max=1):
    return np.interp(newimage, (img.min(), img.max()), (min, max))

def compute_histogram_range(min_range, range=[0, 1]):
    v1 = random.uniform(range[0], range[1])
    v2 = random.uniform(range[0], range[1])
    vmin = min(v1, v2)
    vmax = max(v1, v2)
    if vmax-vmin<min_range:
        vmin = max(range[0], vmin-min_range/2.0)
        vmax = vmin + min_range
    return vmin, vmax

def random_histogram_range(img, min_range=0.1, range=[0,1]):
    min, max = compute_histogram_range(min_range=, range)
    return adjust_histogram_range(img, min, max)

def add_gaussian_noise(img, mean=0, sigma=[0, 0.1]):
    if is_list(sigma):
        if len(sigma)==2:
            sigma = random.uniform(sigma[0], sigma[1])
        else:
            raise ValueError("Sigma  should be either a list/tuple of lenth 2 or a scalar")
    gauss = np.random.normal(mean,sigma,img.shape).reshape(img.shape)
    return img + gauss

def add_speckle_noise(img): # todo: test
    gauss = np.random.randn(img.shape).reshape(img.shape)
    return image + image * gauss

def add_poisson_noise(img): # todo: test
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(image * vals) / float(vals)
    return noisy

def is_list(l):
    return isinstance(l, (list, tuple, np.ndarray))

import skimage.transform as trans
from scipy import interpolate

def zoomshift(I,zoomlevel,shiftX,shiftY, order=0): # todo: test
    '''
    This function applies a zooming/scaling operation on the image, shifts it,
    and then crops it back to its original size
    From delta software: https://gitlab.com/dunloplab/delta/blob/master/data.py
    '''

    oldshape = I.shape
    I = trans.rescale(I,zoomlevel,mode='edge',multichannel=False, order=order)
    shiftX = shiftX * I.shape[0]
    shiftY = shiftY * I.shape[1]
    I = shift(I,(shiftY, shiftX),order=order) # For some reason it looks like X & Y are inverted?
    i0 = (round(I.shape[0]/2 - oldshape[0]/2), round(I.shape[1]/2 - oldshape[1]/2))
    I = I[i0[0]:(i0[0]+oldshape[0]), i0[1]:(i0[1]+oldshape[1])]
    return I

def shift(image, vector, order=0): # todo: test
    '''
    This function performs the shifting operation used in zoomshift() above
    From delta software: https://gitlab.com/dunloplab/delta/blob/master/data.py
    '''
    transform = trans.AffineTransform(translation=vector)
    shifted = trans.warp(image, transform, mode='edge',order=order)

    return shifted

def histogram_voodoo(image,num_control_points=3): # todo: test
    '''
    From delta software: https://gitlab.com/dunloplab/delta/blob/master/data.py
    It performs an elastic deformation on the image histogram to simulate
    changes in illumination
    '''
    control_points = np.linspace(0,1,num=num_control_points+2)
    sorted_points = copy.copy(control_points)
    random_points = np.random.uniform(low=0.1,high=0.9,size=num_control_points)
    sorted_points[1:-1] = np.sort(random_points)
    mapping = interpolate.PchipInterpolator(control_points, sorted_points)

    return mapping(image)

def illumination_voodoo(image,num_control_points=5): # todo: test
    '''
    From delta software: https://gitlab.com/dunloplab/delta/blob/master/data.py
    It simulates a variation in illumination along the length of the chamber
    '''

    # Create a random curve along the length of the chamber:
    control_points = np.linspace(0,image.shape[0]-1,num=num_control_points)
    random_points = np.random.uniform(low=0.1,high=0.9,size=num_control_points)
    mapping = interpolate.PchipInterpolator(control_points, random_points)
    curve = mapping(np.linspace(0,image.shape[0]-1,image.shape[0]))
    # Apply this curve to the image intensity along the length of the chamebr:
    newimage = np.multiply(image,
                            np.reshape(
                                np.tile(
                                    np.reshape(curve,curve.shape + (1,)), (1, image.shape[1])) ,image.shape ) )
    # Rescale values to original range:
    newimage = np.interp(newimage, (newimage.min(), newimage.max()), (image.min(), image.max()))

    return newimage
