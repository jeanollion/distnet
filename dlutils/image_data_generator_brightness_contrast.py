from keras_preprocessing.image import ImageDataGenerator
from .pre_processing_utils import compute_random_brightness_contrast, adjust_brightness_contrast

class ImageDataGeneratorBC(ImageDataGenerator):
    def __init__(self, brightness_range_=None, contrast_range=None, invert=False, **kwargs):
        super().__init__(**kwargs)
        self.brightness_range_=brightness_range_
        self.contrast_range=contrast_range
        self.invert = invert

    def get_random_transform(self, img_shape, seed=None):
        params = super().get_random_transform(img_shape, seed)
        b, c = compute_random_brightness_contrast(self.brightness_range_, self.contrast_range, self.invert)
        params['contrast']  = c
        params['brightness_'] = b
        return params

    def apply_transform(self, x, transform_parameters):
        x = super().apply_transform(x, transform_parameters)
        b = transform_parameters.get("brightness_", 0)
        c = transform_parameters.get("contrast", 1)
        return adjust_brightness_contrast(x, b, c)
