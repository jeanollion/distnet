from dlutils import MultiChannelIterator
import numpy as np

class UnetIterator(MultiChannelIterator):
    def __init__(self,
        h5py_file_path,
        channel_keywords=['/raw', '/regionLabels', '/weightMap'],
        input_channels=[0],
        output_channels=[1, 2],
        mask_channels=[1],
        output_multiplicity = 1,
        channel_scaling_param = None,
        group_keyword=None,
        image_data_generators=None,
        batch_size=32,
        shuffle=True,
        perform_data_augmentation=True,
        seed=None,
        dtype='float32'):
        assert len(channel_keywords)>=3, "At least 3 channels: raw, labels and weights, in this order"
        assert 0 in input_channels
        assert 1 in output_channels
        assert 2 in output_channels
        assert 1 in mask_channels
        super().__init__(h5py_file_path, channel_keywords, input_channels, output_channels, mask_channels, output_multiplicity, channel_scaling_param, group_keyword, image_data_generators, batch_size, shuffle, perform_data_augmentation, seed, dtype)

    def _get_output_batch(self, batch_by_channel, ref_chan_idx, aug_param_array):
        # just merge labels and weight
        labels = batch_by_channel[1]
        weights = batch_by_channel[2]
        return np.concatenate([labels, weights], axis=-1)
