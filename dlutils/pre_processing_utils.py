import numpy as np
from scipy.ndimage.filters import gaussian_filter
import random
from random import uniform, random, randint, getrandbits

def sometimes(func, prob=0.5):
    return lambda im:func(im) if random()<prob else im

def apply_successively(*functions):
    if len(functions)==0:
        return lambda img:img
    def func(img):
        for f in functions:
            img = f(img)
        return img
    return func

def gaussian_blur(img, sig):
    if len(img.shape)>2 and img.shape[-1]==1:
        return np.expand_dims(gaussian_filter(img.squeeze(-1), sig), -1)
    else:
        return gaussian_filter(img, sig)

def random_gaussian_blur(img, sig_min=1, sig_max=2):
    sig = uniform(sig_min, sig_max)
    return gaussian_blur(img, sig)

def adjust_histogram_range(img, min=0, max=1):
    return np.interp(img, (img.min(), img.max()), (min, max))

def compute_histogram_range(min_range, range=[0, 1]):
    if range[1]-range[0]<min_range:
        raise ValueError("Range must be superior to min_range")
    vmin = uniform(range[0], range[1]-min_range)
    vmax = uniform(vmin+min_range, range[1])
    return vmin, vmax

def random_histogram_range(img, min_range=0.1, range=[0,1]):
    min, max = compute_histogram_range(min_range, range)
    return adjust_histogram_range(img, min, max)

def add_gaussian_noise(img, mean=0, sigma=[0, 0.1]):
    if is_list(sigma):
        if len(sigma)==2:
            sigma = uniform(sigma[0], sigma[1])
        else:
            raise ValueError("Sigma  should be either a list/tuple of lenth 2 or a scalar")
    sigma *= (img.max() - img.min())
    gauss = np.random.normal(mean,sigma,img.shape).reshape(img.shape)
    return img + gauss

def add_speckle_noise(img, mean=0, sigma=[0, 0.1]):
    if is_list(sigma):
        if len(sigma)==2:
            sigma = uniform(sigma[0], sigma[1])
        else:
            raise ValueError("Sigma  should be either a list/tuple of lenth 2 or a scalar")
    gauss = np.random.normal(mean, sigma, img.shape).reshape(img.shape)
    return img + img * gauss

def add_poisson_noise(img, noise_intensity=[0, 0.1]):
    if is_list(noise_intensity):
        if len(noise_intensity)==2:
            noise_intensity = uniform(noise_intensity[0], noise_intensity[1])
        else:
            raise ValueError("noise_intensity should be either a list/tuple of lenth 2 or a scalar")
    noise_intensity /= 10.0 # so that intensity is comparable to gaussian sigma
    min = img.min()
    max = img.max()
    img = (img - min) / (max - min)
    output = np.random.poisson(img / noise_intensity) * noise_intensity
    return output * (max - min) + min

def noise_function(noise_max_intensity=0.1):
    def res(img):
        gauss = not getrandbits(1)
        speckle = not getrandbits(1)
        poisson = not getrandbits(1)
        ni = noise_max_intensity / float(max(1, sum([gauss, speckle, poisson])))
        funcs = []
        if poisson:
            funcs.append(lambda im : add_poisson_noise(im, noise_intensity=[0, ni]))
        if speckle:
            funcs.append(lambda im:add_speckle_noise(im, sigma=[0, ni]))
        if gauss:
            funcs.append(lambda im:add_gaussian_noise(im, sigma=[0, ni]))
        return apply_successively(*funcs)(img)
    return res

def grayscale_deformation_function(add_noise=True, adjust_histogram_range=True):
    funcs = [sometimes(random_gaussian_blur), sometimes(histogram_voodoo), sometimes(illumination_voodoo)]
    if add_noise:
        funcs.append(noise_function())
    if adjust_histogram_range:
        funcs.append(lambda img:random_histogram_range(img))
    return lambda img:apply_successively(*funcs)(img)

def is_list(l):
    return isinstance(l, (list, tuple, np.ndarray))





# import skimage.transform as trans
# def zoomshift(I,zoomlevel,shiftX,shiftY, order=0): # todo: test
#     '''
#     This function applies a zooming/scaling operation on the image, shifts it,
#     and then crops it back to its original size
#     From delta software: https://gitlab.com/dunloplab/delta/blob/master/data.py
#     '''
#
#     oldshape = I.shape
#     I = trans.rescale(I,zoomlevel,mode='edge',multichannel=False, order=order)
#     shiftX = shiftX * I.shape[0]
#     shiftY = shiftY * I.shape[1]
#     I = shift(I,(shiftY, shiftX),order=order) # For some reason it looks like X & Y are inverted?
#     i0 = (round(I.shape[0]/2 - oldshape[0]/2), round(I.shape[1]/2 - oldshape[1]/2))
#     I = I[i0[0]:(i0[0]+oldshape[0]), i0[1]:(i0[1]+oldshape[1])]
#     return I
#
# def shift(image, vector, order=0): # todo: test
#     '''
#     This function performs the shifting operation used in zoomshift() above
#     From delta software: https://gitlab.com/dunloplab/delta/blob/master/data.py
#     '''
#     transform = trans.AffineTransform(translation=vector)
#     shifted = trans.warp(image, transform, mode='edge',order=order)
#
#     return shifted

from scipy import interpolate
import copy

def histogram_voodoo(image,num_control_points=50):
    '''
    Adapted from delta software: https://gitlab.com/dunloplab/delta/blob/master/data.py
    It performs an elastic deformation on the image histogram to simulate
    changes in illumination
    '''
    min = image.min()
    max = image.max()
    range = max - min
    delta = range/float(num_control_points+2)
    control_points = np.linspace(min,max,num=num_control_points+2)
    sorted_points = copy.copy(control_points)
    random_points = np.random.uniform(low=min + delta, high=max - delta,size=num_control_points)
    sorted_points[1:-1] = np.sort(random_points)
    mapping = interpolate.PchipInterpolator(control_points, sorted_points)

    newimage = mapping(image)
    # Rescale values to original range:
    #newimage = np.interp(newimage, (newimage.min(), newimage.max()), (image.min(), image.max()))
    return newimage

def illumination_voodoo(image,num_control_points=5, intensity=0.6):
    '''
    Adapted from delta software: https://gitlab.com/dunloplab/delta/blob/master/data.py
    It simulates a variation in illumination along the length of the chamber
    '''
    if intensity>=1 or intensity<=0:
        raise ValueError("Intensity should be in range ]0, 1[")
    # Create a random curve along the length of the chamber:
    control_points = np.linspace(0,image.shape[0]-1,num=num_control_points)
    random_points = np.random.uniform(low=(1 - intensity) / 2.0, high=(1 + intensity) / 2.0, size=num_control_points)
    mapping = interpolate.PchipInterpolator(control_points, random_points)
    curve = mapping(np.linspace(0,image.shape[0]-1,image.shape[0]))
    curveIm = np.reshape( np.tile( np.reshape(curve,curve.shape + (1,)), (1, image.shape[1])) ,image.shape )
    # Apply this curve to the image intensity along the length of the chamebr:
    newimage = np.multiply(image, curveIm)
    # Rescale values to original range:
    newimage = np.interp(newimage, (newimage.min(), newimage.max()), (image.min(), image.max()))

    return newimage
