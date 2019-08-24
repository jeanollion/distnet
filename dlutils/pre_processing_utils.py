import numpy as np
from scipy.ndimage.filters import gaussian_filter
import random

def sometimes(func, prob=0.5):
    return lambda im:func(im) if random.random()<prob else im

def gaussian_blur(img, sig):
    return np.expand_dims(gaussian_filter(img.squeeze(-1), sig), -1)

def random_gaussian_blur(img, sig_min=1, sig_max=2):
    sig = random.uniform(sig_min, sig_max)
    return np.expand_dims(gaussian_filter(img.squeeze(-1), sig), -1)
