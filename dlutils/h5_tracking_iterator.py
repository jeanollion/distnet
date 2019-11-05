import numpy as np
from dlutils import H5SegmentationIterator
from .utils import get_datasets_by_path
from random import random
from sklearn.model_selection import train_test_split
from .h5_multichannel_iterator import copy_geom_tranform_parameters

class H5TrackingIterator(H5SegmentationIterator):
	def __init__(self,
				h5py_file_path,
				channel_keywords=['/raw', '/regionLabels', '/prevRegionLabels', '/edm'],
				input_channels=[0, 1],
				output_channels=[2],
				input_channels_prev=[True, True],
				input_channels_next=[False, False],
				output_channels_prev=[False],
				output_channels_next=[False],
				mask_channel=1,
				channel_scaling_param=None, #[{'level':1, 'qmin':5, 'qmax':95}],
				group_keyword=None,
				image_data_generators=None,
				batch_size=32,
				shuffle=True,
				perform_data_augmentation=True,
				seed=None,
				dtype='float32'):

		if len(input_channels_next)!=len(input_channels):
			raise ValueError("length of input_channels_next differs from input_channels")
		if len(input_channels_prev)!=len(input_channels):
			raise ValueError("length of input_channels_prev differs from input_channels")
		if len(output_channels_next)!=len(output_channels):
			raise ValueError("length of input_channels_next differs from output_channels")
		if len(output_channels_prev)!=len(output_channels):
			raise ValueError("length of input_channels_prev differs from output_channels")

		if (any(input_channels_prev) or any(output_channels_prev)) and not input_channels_prev[0]:
			raise ValueError("Previous time point of first channel should be returned if previous time point from another channel is returned")
		if (any(input_channels_next) or any(output_channels_next)) and not input_channels_next[0]:
			raise ValueError("Next time point of first channel should be returned if next time point from another channel is returned")

		self.input_channels_prev=input_channels_prev
		self.input_channels_next=input_channels_next
		self.output_channels_prev=output_channels_prev
		self.output_channels_next=output_channels_next
		super().__init__(h5py_file_path, channel_keywords, input_channels, output_channels, mask_channel, channel_scaling_param, group_keyword, image_data_generators, batch_size, shuffle, perform_data_augmentation, seed)

	def _include_next(self, chan_idx, is_input):
		if is_input:
			if chan_idx not in self.input_channels:
				return False
			idx = self.input_channels.index(chan_idx)
			return self.input_channels_next[idx]
		else:
			if chan_idx not in self.output_channels:
				return False
			idx = self.output_channels.index(chan_idx)
			return self.output_channels_next[idx]

	def _include_prev(self, chan_idx, is_input):
		if is_input:
			if chan_idx not in self.input_channels:
				return False
			idx = self.input_channels.index(chan_idx)
			return self.input_channels_prev[idx]
		else:
			if chan_idx not in self.output_channels:
				return False
			idx = self.output_channels.index(chan_idx)
			return self.output_channels_prev[idx]

	def _get_batches_of_transformed_samples_by_channel(self, index_ds, index_array, chan_idx, is_input, aug_param_array=None, perform_augmentation=True):
		if is_input and chan_idx==0: # signal indices without neighbor time point (absent or result of data augmentation)
			if self._include_prev(0, True):
				self._set_has_neigh_parameters(index_ds, index_array, aug_param_array, True, perform_augmentation)
			if self._include_next(0, True):
				self._set_has_neigh_parameters(index_ds, index_array, aug_param_array, False, perform_augmentation)

		def transfer_aug_param_function(source, dest, ds_idx, im_idx):
			if len(source) <= 2 and ("no_prev" in source or "no_next" in source or "no_prev_aug" in source or "no_next_aug" in source): # special case: channel 0 input, augmentation parameter was initiated to signal no prev / no next: parameters are actually contained in dest
				for k in source.keys(): # copy prev / next signaling needed to limit augmentation parameters
					dest[k] = source[k]
				if dest.get('zx', 1)>1: # forbid zx>1 if there are object @ border @ prev/next time point because zoom will be copied to prev/next transformation
					if "no_prev" in source and not source['no_prev']:
						has_object_up, has_object_down = self._has_object_at_y_borders(self.mask_channel, ds_idx, im_idx - 1)
						if has_object_up or has_object_down:
							dest["zx"] = 1
					elif "no_next" in source and not source['no_next']:
						has_object_up, has_object_down = self._has_object_at_y_borders(self.mask_channel, ds_idx, im_idx + 1)
						if has_object_up or has_object_down:
							dest["zx"] = 1
				# also copy all element from dest to source, because only source will remain in aug_param_array
				for k in dest.keys():
					source[k] = dest[k]

			else:
				copy_geom_tranform_parameters(source, dest)
				if "aug_params_prev" in source:
					dest["aug_params_prev"] = dict()
					copy_geom_tranform_parameters(source["aug_params_prev"], dest["aug_params_prev"])
				if "aug_params_next" in source:
					dest["aug_params_next"] = dict()
					copy_geom_tranform_parameters(source["aug_params_next"], dest["aug_params_next"])

		batch = super()._get_batches_of_transformed_samples_by_channel(index_ds, index_array, chan_idx, is_input, aug_param_array, perform_augmentation, transfer_aug_param_function)
		batch_prev = self._get_batches_of_transformed_samples_by_channel_neighbor(index_ds, index_array, chan_idx, is_input, aug_param_array, True, perform_augmentation) if self._include_prev(chan_idx, is_input) else None
		batch_next = self._get_batches_of_transformed_samples_by_channel_neighbor(index_ds, index_array, chan_idx, is_input, aug_param_array, False, perform_augmentation) if self._include_next(chan_idx, is_input) else None
		#print("channel {} (prev: {}, next: {}) parameters: {}".format(chan_idx, self._include_prev(chan_idx, is_input), self._include_next(chan_idx, is_input), aug_param_array[0][chan_idx]))
		if batch_prev is not None and batch_next is not None:
			return np.concatenate((batch_prev, batch, batch_next), axis=-1)
		elif batch_prev is not None:
			return np.concatenate((batch_prev, batch), axis=-1)
		elif batch_next is not None:
			return np.concatenate((batch, batch_next), axis=-1)
		else:
			return batch

	def _set_has_neigh_parameters(self, index_ds, index_array, aug_param_array, prev , perform_augmentation=True, aug_remove_prob=0.1):
		inc = -1 if prev else 1
		no_neigh_key = "no_prev" if prev else "no_next"
		no_neigh_aug_key = "no_prev_aug" if prev else "no_next_aug"

		for i, (ds_idx, im_idx) in enumerate(zip(index_ds, index_array)):
			neigh_lab = get_neighbor_label(self.labels[ds_idx][im_idx], prev=prev)
			bound_idx = 0 if prev else len(self.labels[ds_idx])-1
			if aug_param_array[i][0] is None:
				aug_param_array[i][0] = dict()
			if im_idx==bound_idx or neigh_lab!=self.labels[ds_idx][im_idx+inc]: # no neighbor image + signal in order to erase displacement map in further steps
				aug_param_array[i][0][no_neigh_key] = True
			elif self.perform_data_augmentation and perform_augmentation and random() < aug_remove_prob: # neighbor image is erased as part of data augmentation + signal in order to set constant displacement map in further steps
				aug_param_array[i][0][no_neigh_aug_key] = True
			else:
				aug_param_array[i][0][no_neigh_key] = False # signal that prev / next will be fetched

	def _get_batches_of_transformed_samples_by_channel_neighbor(self, index_ds, index_array, chan_idx, is_input, aug_param_array, prev, perform_augmentation=True):
		inc = -1 if prev else 1
		no_neigh_key = "no_prev" if prev else "no_next"
		no_neigh_aug_key = "no_prev_aug" if prev else "no_next_aug"
		aug_key = "aug_params_prev" if prev else "aug_params_next"
		dy_shift_key = 'dy_shift_prev' if prev else 'dy_shift_next'

		def transfer_aug_param_function(source, dest, ds_idx, im_idx):
			self._transfer_illumination_aug_param(source, dest)
			if aug_key not in source: # geom transformation are constrained by parameters of current time point
				self._transfer_geom_aug_param_neighbor(source, dest, prev, ds_idx, im_idx)
				source[aug_key] = dest  # also stores the paramaters in the source for next channels
				source[dy_shift_key] = (source.get('tx', 0) - dest.get('tx', 0)) * inc # also store dy shift (# todo source/dest inverted ? )
			else: # geom parameters are copied from neighbor time point of the first augmented channel
				copy_geom_tranform_parameters(source[aug_key], dest)

		# define new index array. no prev / no next tag is in the aug_param_array
		index_array = np.copy(index_array)
		index_array += inc
		missing_idx = [i for i in range(len(aug_param_array)) if aug_param_array[i][0].get(no_neigh_key, False) or aug_param_array[i][0].get(no_neigh_aug_key, False)]
		index_array[missing_idx] -= inc
		return super()._get_batches_of_transformed_samples_by_channel(index_ds, index_array, chan_idx, is_input, aug_param_array, perform_augmentation, transfer_aug_param_function)

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

	def _transfer_geom_aug_param_neighbor(self, source, dest, prev, ds_idx, im_idx): # transfer affine parameters that must be identical between curent and prev/next image
		dest['flip_vertical'] = source.get('flip_vertical', False) # flip must be the same
		dest['zy'] = source.get('zy', 1) # zoom should be the same so that cell aspect does not change too much
		dest['zx'] = source.get('zx', 1) # zoom should be the same so that cell aspect does not change too much
		dest['shear'] = source.get('shear', 0) # shear should be the same so that cell aspect does not change too much
		self._forbid_transformations_if_object_touching_borders(dest, self.mask_channel, ds_idx, im_idx) # im_idx are already those of prev/next image

	def _transfer_illumination_aug_param(self, source, dest):
		# illumination parameters should be the same between current and neighbor images
		if 'vmin' in source:
			dest['vmin'] = source['vmin']
		elif 'vmin' in dest:
			del dest['vmin']
		if 'vmax' in source:
			dest['vmax'] = source['vmax']
		elif 'vmax' in dest:
			del dest['vmax']
		if 'brightness' in source:
			dest['brightness'] = source['brightness']
		elif 'brightness' in dest:
			del dest['brightness']

# class util methods
def get_neighbor_label(label, prev):
	frame = int(label[-5:])
	if prev and frame==0:
		return None
	return label[:-5]+str(frame+(-1 if prev else 1)).zfill(5)
