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

def adjust_brightness_contrast(img, brightness=0, contrast=1):
    if brightness!=0 and contrast!=1:
        return img * contrast + brightness
    elif brightness!=0:
        return img + brightness
    elif contrast!=1:
        return img * contrast
    else:
        return img

def random_brightness_contrast(img, brightness_range=[-0.5, 0.5], contrast_range=1.5, invert=True):
    b, c = compute_random_brightness_contrast(brightness_range, contrast_range, invert)
    return adjust_brightness_contrast(img, b, c)

def compute_random_brightness_contrast(brightness_range=[-0.5, 0.5], contrast_range=1.5, invert=True):
    if brightness_range:
        if is_list(brightness_range):
            if len(brightness_range)!=2:
                raise ValueError("brightness_range should be either a list/tuple of length 2 or a scalar or None")
        else:
            brightness_range = (-abs(brightness_range), abs(brightness_range))
    if contrast_range:
        if is_list(contrast_range):
            if len(contrast_range)!=2:
                raise ValueError("contrast_range should be either a list/tuple of length 2 or a scalar or None")
        else:
            if abs(contrast_range)>1:
                contrast_range = (1/abs(contrast_range), abs(contrast_range))
            else:
                contrast_range = (abs(contrast_range), 1/abs(contrast_range))

    b = random.uniform(brightness_range[0], brightness_range[1]) if brightness_range else 0
    c = random.uniform(contrast_range[0], contrast_range[1]) if contrast_range else 0
    if invert and bool(random.getrandbits(1)):
        c = -c
    return (b,c)

def add_gaussian_noise(img, mean=0, sigma=[0, 0.1]):
    if is_list(sigma):
        if len(sigma)==2:
            sigma = random.uniform(sigma[0], sigma[1])
        else:
            raise ValueError("Sigma  should be either a list/tuple of lenth 2 or a scalar")
    gauss = np.random.normal(mean,sigma,img.shape).reshape(img.shape)
    return img + gauss

def is_list(l):
    return isinstance(l, (list, tuple, np.ndarray))
