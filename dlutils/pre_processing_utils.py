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

def random_brightness_contrast(img, brightness_range=[-0.5, 0.5], contrast_range=1.5, invert=True):
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
    if brightness_range and contrast_range:
        b = random.uniform(brightness_range[0], brightness_range[1])
        c = random.uniform(contrast_range[0], contrast_range[1])
        if invert and bool(random.getrandbits(1)):
            c = - c
        return img * c + b
    elif brightness_range:
        b = random.uniform(brightness_range[0], brightness_range[1])
        if invert and bool(random.getrandbits(1)):
            return -img + b
        else:
            return img + b
    elif contrast_range:
        c = random.uniform(contrast_range[0], contrast_range[1])
        if invert and bool(random.getrandbits(1)):
            c = - c
        return img * c
    else:
        return img

def is_list(l):
    return isinstance(l, (list, tuple, np.ndarray))
