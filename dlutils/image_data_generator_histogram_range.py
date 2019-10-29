from keras_preprocessing.image import ImageDataGenerator
from .pre_processing_utils import compute_histogram_range, adjust_histogram_range
from random import random

class ImageDataGeneratorHistogramRange(ImageDataGenerator):
    def __init__(self, histogram_min_range=0.1, histogram_range=[0,1], invert_prob=0, **kwargs):
        super().__init__(**kwargs)
        self.histogram_min_range=histogram_min_range
        self.histogram_range = histogram_range
        self.invert_prob = invert_prob

    def get_random_transform(self, img_shape, seed=None):
        params = super().get_random_transform(img_shape, seed)
        vmin, vmax = compute_histogram_range(self.histogram_min_range, self.histogram_range)
        params['vmin'] = vmin
        params['vmax'] = vmax
        if self.invert_prob>0 and (self.invert_prob==1 or random()<self.invert_prob):
            temp = vmax
            vmax = vmin
            vmin = temp
        return params

    def apply_transform(self, x, transform_parameters):
        x = super().apply_transform(x, transform_parameters)
        vmin = transform_parameters.get("vmin", self.histogram_range[0])
        vmax = transform_parameters.get("vmax", self.histogram_range[1])
        return adjust_histogram_range(x, vmin, vmax)
