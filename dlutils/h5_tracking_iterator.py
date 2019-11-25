import numpy as np
from dlutils import H5MultiChannelIterator
from .utils import get_datasets_by_path
from random import random
from sklearn.model_selection import train_test_split
from .h5_multichannel_iterator import copy_geom_tranform_parameters
from dlutils.image_data_generator_mm import transfer_illumination_aug_parameters

class H5TrackingIterator(H5MultiChannelIterator):
	def __init__(self,
				h5py_file_path,
				channel_keywords=['/raw', '/regionLabels', '/prevRegionLabels', '/edm'],
				input_channels=[0, 1],
				output_channels=[2],
				channels_prev=[True, True, True, True],
				channels_next=[False, False, False, False],
				mask_channels=[1, 2, 3],
				channel_scaling_param=None, #[{'level':1, 'qmin':5, 'qmax':95}],
				group_keyword=None,
				image_data_generators=None,
				batch_size=32,
				shuffle=True,
				perform_data_augmentation=True,
				seed=None,
				dtype='float32'):

		if len(channels_next)!=len(channel_keywords):
			raise ValueError("length of channels_next differs from channel_keywords")
		if len(channels_prev)!=len(channel_keywords):
			raise ValueError("length of channels_prev differs from channel_keywords")

		if any(channels_prev) and not channels_prev[mask_channels[0]]:
			raise ValueError("Previous time point of first mask channel should be returned if previous time point from another channel is returned")
		if any(channels_next) and not channels_next[mask_channels[0]]:
			raise ValueError("Next time point of first mask channel should be returned if next time point from another channel is returned")

		self.channels_prev=channels_prev
		self.channels_next=channels_next
		self.aug_remove_prob = 0.05 # set current image as prev / next
		super().__init__(h5py_file_path, channel_keywords, input_channels, output_channels, mask_channels, channel_scaling_param, group_keyword, image_data_generators, batch_size, shuffle, perform_data_augmentation, seed)

	def _get_batches_of_transformed_samples_by_channel(self, index_ds, index_array, chan_idx, ref_chan_idx, aug_param_array=None, perform_augmentation=True):
		def transfer_aug_param_function(source, dest): # also copies prev/next
			copy_geom_tranform_parameters(source, dest)
			if "aug_params_prev" in source:
				if "aug_params_prev" not in dest:
					dest["aug_params_prev"] = dict()
				copy_geom_tranform_parameters(source["aug_params_prev"], dest["aug_params_prev"])
			if "aug_params_next" in source:
				if "aug_params_next" not in dest:
					dest["aug_params_next"] = dict()
				copy_geom_tranform_parameters(source["aug_params_next"], dest["aug_params_next"])
		return super()._get_batches_of_transformed_samples_by_channel(index_ds, index_array, chan_idx, ref_chan_idx, aug_param_array, perform_augmentation, transfer_aug_param_function=transfer_aug_param_function)

	def _apply_augmentation(self, img, chan_idx, aug_params): # apply separately for prev / cur / next
		if "aug_params_prev" in aug_params and self.channels_prev[chan_idx]:
			img[...,0:1] = super()._apply_augmentation(img[...,0:1], chan_idx, aug_params.get("aug_params_prev"))
		if "aug_params_next" in aug_params and self.channels_next[chan_idx]:
			img[...,-1:0] = super()._apply_augmentation(img[...,-1:0], chan_idx, aug_params.get("aug_params_next"))
		cur_chan_idx = 1 if self.channels_prev[chan_idx] else 0
		img[...,cur_chan_idx:(cur_chan_idx+1)] = super()._apply_augmentation(img[...,cur_chan_idx:(cur_chan_idx+1)], chan_idx, aug_params)
		return img

	def _get_data_augmentation_parameters(self, chan_idx, ref_chan_idx, batch, idx, index_ds, index_array):
		batch_chan_idx = 1 if self.channels_prev[chan_idx] else 0
		params = super()._get_data_augmentation_parameters(chan_idx, ref_chan_idx, batch[...,batch_chan_idx:(batch_chan_idx+1)], idx, index_ds, index_array)
		if chan_idx==ref_chan_idx and chan_idx in self.mask_channels:
			if self.channels_prev[chan_idx] :
				try:
					self.image_data_generators[chan_idx].adjust_augmentation_param_from_neighbor_mask(params, batch[idx,...,0])
				except AttributeError: # data generator does not have this method
					pass
			if self.channels_next[chan_idx]:
				try:
					self.image_data_generators[chan_idx].adjust_augmentation_param_from_neighbor_mask(params, batch[idx,...,-1])
				except AttributeError: # data generator does not have this method
					pass
		if self.channels_prev[chan_idx]:
			params_prev = super()._get_data_augmentation_parameters(chan_idx, ref_chan_idx, batch[...,0:1], idx, index_ds, index_array)
			self._transfer_illumination_aug_param(params, params_prev)
			self._transfer_geom_aug_param_neighbor(params, params_prev)
			try:
				self.image_data_generators[chan_idx].adjust_augmentation_param_from_mask(params_prev, batch[idx,...,0])
			except AttributeError: # data generator does not have this method
				pass
			params["aug_params_prev"] = params_prev
		if self.channels_next[chan_idx]:
			params_next = super()._get_data_augmentation_parameters(chan_idx, ref_chan_idx, batch[...,-1:0], idx, index_ds, index_array)
			self._transfer_illumination_aug_param(params, params_next)
			self._transfer_geom_aug_param_neighbor(params, params_next)
			try:
				self.image_data_generators[chan_idx].adjust_augmentation_param_from_mask(params_next, batch[idx,...,-1])
			except AttributeError: # data generator does not have this method
				pass
			params["aug_params_next"] = params_next
		return params

	def _transfer_geom_aug_param_neighbor(self, source, dest): # transfer affine parameters that must be identical between curent and prev/next image
		dest['flip_vertical'] = source.get('flip_vertical', False) # flip must be the same
		dest['zy'] = source.get('zy', 1) # zoom should be the same so that cell aspect does not change too much
		dest['zx'] = source.get('zx', 1) # zoom should be the same so that cell aspect does not change too much
		dest['shear'] = source.get('shear', 0) # shear should be the same so that cell aspect does not change too much

	def _transfer_illumination_aug_param(self, source, dest):
		# illumination parameters should be the same between current and neighbor images
		transfer_illumination_aug_parameters(source, dest)
		if 'brightness' in source:
			dest['brightness'] = source['brightness']
		elif 'brightness' in dest:
			del dest['brightness']

	def _read_image_batch(self, index_ds, index_array, chan_idx, ref_chan_idx, aug_param_array):
		batch = super()._read_image_batch(index_ds, index_array, chan_idx, ref_chan_idx, aug_param_array)
		batch_prev = self._read_image_batch_neigh(index_ds, index_array, chan_idx, ref_chan_idx, True, aug_param_array) if self.channels_prev[chan_idx] else None
		batch_next = self._read_image_batch_neigh(index_ds, index_array, chan_idx, ref_chan_idx, False, aug_param_array) if self.channels_prev[chan_idx] else None
		if batch_prev is not None and batch_next is not None:
			return np.concatenate((batch_prev, batch, batch_next), axis=-1)
		elif batch_prev is not None:
			return np.concatenate((batch_prev, batch), axis=-1)
		elif batch_next is not None:
			return np.concatenate((batch, batch_next), axis=-1)
		else:
			return batch

	def _read_image_batch_neigh(self, index_ds, index_array, chan_idx, ref_chan_idx, prev, aug_param_array):
		no_neigh = 'no_prev' if prev else 'no_next'
		inc = -1 if prev else 1
		if chan_idx==ref_chan_idx: # flag that there is no neighbor in aug_param_array for further computation
			for i, (ds_idx, im_idx) in enumerate(zip(index_ds, index_array)):
				neigh_lab = get_neighbor_label(self.labels[ds_idx][im_idx], prev=prev)
				bound_idx = 0 if prev else len(self.labels[ds_idx]) - 1
				if im_idx==bound_idx or neigh_lab!=self.labels[ds_idx][im_idx+inc]: # no neighbor image + signal in order to erase displacement map in further steps
					aug_param_array[i][ref_chan_idx][no_neigh] = True
				elif self.perform_data_augmentation and aug_param_array is not None and random() < self.aug_remove_prob: # neighbor image is replaced by current image as part of data augmentation + signal in order to set constant displacement map in further steps
					aug_param_array[i][ref_chan_idx][no_neigh] = True
				else:
					aug_param_array[i][ref_chan_idx][no_neigh] = False

		index_array = np.copy(index_array)
		index_array += inc
		for i in range(len(index_ds)): # missing images -> set current image instead.
			if aug_param_array[i][ref_chan_idx][no_neigh]:
				index_array[i] -= inc
		return super()._read_image_batch(index_ds, index_array, chan_idx, ref_chan_idx, aug_param_array)

	def train_test_split(self, **options):
		train_iterator, test_iterator = super().train_test_split(**options)
		train_idx = train_iterator.allowed_indexes
		test_idx = test_iterator.allowed_indexes
		# remove neighboring time points that are seen by the network. only in terms of ground truth, ie depends on returned values:  previous and next frames or next frame only (displacement)
		if any(self.output_channels_prev) or any(self.input_channels_prev): # an index visited in train_idx implies the previous one is also seen during training. to avoind that previous index being in test_idx, next indices of test_idx should remove from train_idx
			train_idx = np.setdiff1d(train_idx, self._get_neighbor_indices(test_idx, prev=False))
		if any(self.output_channels_next) or any(self.input_channels_next): # an index visited in train_idx implies the next one is also seen during training. to avoin that next index being in test_idx, previous indices of test_idx should remove from train_idx
			train_idx = np.setdiff1d(train_idx, self._get_neighbor_indices(test_idx, prev=True))

		train_iterator.set_allowed_indexes(train_idx)

		return train_iterator, test_iterator

	# for train test split
	def _get_neighbor_indices(self, index_array, prev):
		index_array_local = np.copy(index_array)
		ds_idx_array = self._get_ds_idx(index_array_local)
		res = []
		inc = -1 if prev else 1
		for i, (ds_idx, im_idx) in enumerate(zip(ds_idx_array, index_array_local)):
			neigh_lab = get_neighbor_label(self.labels[ds_idx][im_idx], prev=prev)
			bound_idx = 0 if prev else len(self.labels[ds_idx])-1
			if im_idx!=bound_idx and neigh_lab==self.labels[ds_idx][im_idx+inc]:
				res.append(index_array[i]+inc)
		return res

# class util methods
def get_neighbor_label(label, prev):
	frame = int(label[-5:])
	if prev and frame==0:
		return None
	return label[:-5]+str(frame+(-1 if prev else 1)).zfill(5)
