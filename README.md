# DistNet: Deep Tracking by displacement regression: application to bacteria growing in the __Mother Machine__

[Link to preprint](https://arxiv.org/abs/2003.07790)

This repo contains sources for:
- Distnet model keras/tensorflow implementation (link to pre-print). Tu use: `disnet.keras_models.get_distnet_model()`
- Flexible keras unet implementation / unet++ implementatoin. See: `disnet.keras_models.unet.py`
- A class of dataset_iterator for training: `distnet.data_generator.dy_iterator.py`
- A class of image data generator with specific transformations for mother machine data. See `distnet.data_generator.image_data_generator_mm.py`
- Utilities for models in keras/tensorflow and image processing. See `distnet.utils.pre_processing.py`
