import numpy as np
from keras_preprocessing.image import Iterator
from .utils import get_datasets_paths, get_datasets_by_path, get_parent_path

class H5Iterator(Iterator):
	def __init__(self,
				h5py_file,
				channel_keywords=['/raw'],
				channel_scaling=[{'level':1, 'qmin':5, 'qmax':95}],
				group_keyword=None,
				image_data_generators=None,
				batch_size=32,
				shuffle=True,
				perform_data_augmentation=True,
				seed=None,
				dtype='float32'):
		self.h5py_file=h5py_file
		self.group_keyword=group_keyword
		self.channel_keywords=channel_keywords
		self.channel_scale_info = channel_scaling
		self.dtype = dtype
		self.perform_data_augmentation=perform_data_augmentation
		if image_data_generators!=None and len(channel_keywords)!=len(image_data_generators):
			raise ValueError('image_data_generators argument should be either None or an array of same length as channel_keywords')
		self.image_data_generators=image_data_generators

		# get all dataset paths
		self.paths = get_datasets_paths(self.h5py_file, channel_keywords[0], self.group_keyword)
		if (len(self.paths)==0):
			raise ValueError('No datasets found ending by {} {}'.format(suffix, "and containing {}".format(group_keyword) if group_keyword!=None else "" ))
		# get all matching dataset lists from h5 file
		self.ds_array = [get_datasets_by_path(self.h5py_file, [self._get_dataset_path(c, ds_idx) for ds_idx in range(len(self.paths))]) for c in range(len(self.channel_keywords))]
		# check that all ds have compatible length between input and output
		indexes = np.array([ds.shape[0] for ds in self.ds_array[0]])
		if len(channel_keywords)>1:
			for c, ds_l in enumerate(self.ds_array):
				if len(self.ds_array[0])!=len(ds_l):
					raise ValueError('Channels {}({}) has #{} datasets whereas first channel has #{} datasets'.format(c, channel_keywords[c], len(ds_l), len(self.ds_array[0])))
				for ds_idx, ds in enumerate(ds_l):
					if indexes[ds_idx] != ds.shape[0]:
						raise ValueError('Channel {}({}) has at least one dataset with number of elements that differ from Channel 0'.format(c, channel_keywords[c]))

		# get offset for each dataset
		for i in range(1, len(indexes)):
			indexes[i]=indexes[i-1]+indexes[i]
		self.ds_len=indexes
		self.ds_off=np.insert(self.ds_len[:-1], 0, 0)

		# check that all datasets have same image shape within each channel
		self.shape = [ds_l[0].shape[1:] for ds_l in self.ds_array]
		for c, ds_l in enumerate(self.ds_array):
			for ds_idx, ds in enumerate(ds_l):
				if ds.shape[1:] != self.shape[c]:
					raise ValueError('Dataset {dsi} with path {dspath} from channel {chan}({chank}) has shape {dsshape} that differs from first dataset with path {ds1path} with shape {ds1shape}'.format(dsi=ds_idx, dspath=self._get_dataset_path(c, ds_idx), chan=c, chank=self.channel_keywords[c], dsshape=ds.shape[1:], ds1path=self._get_dataset_path(c, 0), ds1shape=self.shape[c] ))

		# set scaling information for each dataset
		self.channel_scale = [None]*len(channel_keywords)
		if channel_scaling!=None:
			percentile_x = np.arange(0, 101)
			for c, scaling_info in enumerate(self.channel_scale_info):
				if scaling_info!=None:
					self.channel_scale[c]=[None]*len(self.paths)
					for ds_idx, path in enumerate(self.paths):
						group = get_parent_path(path)
						for i in range(scaling_info.get('level', 1)):
							group = get_parent_path(group)
							if group==None:
								raise ValueError("scaling group level too high for channel {}({}) group path: {}".format(c, channel_keywords[c]), _get_parent_path(path))
						# percentiles are located in attributes of group
						if not self.h5py_file[group].attrs.__contains__('raw_percentiles'):
							raise ValueError("No percentile array found in group {} for channel: {}({]})".format(group, c, channel_keywords[c]))
						percentiles = self.h5py_file[group].attrs.get(channel_keywords[c].replace('/', '')+'_percentiles')
						# get IQR and median
						min, med, max = np.interp([scaling_info.get('qmin', 5), 50, scaling_info.get('qmax', 95)], percentile_x, percentiles)
						self.channel_scale[c][ds_idx] = [med, max-min]

		super(H5Iterator, self).__init__(indexes[-1], batch_size, shuffle, seed) # todo revoir ?


	def _get_batches_of_transformed_samples(self, index_array):
		"""Gets a batch of transformed samples.
		# Arguments
			index_array: Array of sample indices to include in batch.
		# Returns
			A batch of transformed samples (tuple of input and output if output_keyword is specified).
		"""

		ds_idx = np.searchsorted(self.ds_len, index_array, side='right')
		# remove ds offset to each index
		index_array -= self.ds_off[ds_idx]
		if len(self.channel_keywords)==1:
			return self._get_input_batch(ds_idx, index_array)
		elif len(self.channel_keywords)==2 and self.channel_keywords[0]==self.channel_keywords[1]:
			batch = self._get_input_batch(ds_idx, index_array)
			return (batch, batch)
		else:
			aug_param_array = [None]*len(index_array)
			return (self._get_input_batch(ds_idx, index_array, aug_param_array=aug_param_array), self._get_output_batch(ds_idx, index_array, aug_param_array=aug_param_array))

	def _get_input_batch(self, index_ds, index_array, aug_param_array=None):
		batch = self._get_batches_of_transformed_samples_by_channel(index_ds, index_array, 0, True,  aug_param_array)
		if self.image_data_generators!=None and self.image_data_generators[0]!=None:
			for i in range(batch.shape[0]):
				batch[i] = self.image_data_generators[0].standardize(batch[i]) # replace by RobustScaler from quantiles in metadata ?
		return batch

	def _get_output_batch(self, index_ds, index_array, aug_param_array=None):
		return self._get_batches_of_transformed_samples_by_channel(index_ds, index_array, 1, False, aug_param_array)

	def _get_batches_of_transformed_samples_by_channel(self, index_ds, index_array, chan_idx, is_input, aug_param_array=None):
		im_shape = self.shape[chan_idx]
		image_data_generator = self.image_data_generators[chan_idx] if self.perform_data_augmentation and self.image_data_generators!=None else None
		channel = () if len(im_shape)==3 else (1,)
		batch = np.zeros((len(index_array),) + im_shape + channel, dtype=self.dtype)
		# build batch of image data
		for i, (ds_idx, im_idx) in enumerate(zip(index_ds, index_array)):
			im = self._read_image(chan_idx, ds_idx, im_idx)
			if image_data_generator!=None:
				params = image_data_generator.get_random_transform(im.shape)
				if aug_param_array!=None:
					if is_input:
						aug_param_array[i] = params
					else:
						copy_affine_tranform_parameters(aug_param_array[i], params)
				im = image_data_generator.apply_transform(im, params)
			batch[i] = im
		return batch

	def _read_image(self, chan_idx, ds_idx, im_idx):
		ds = self.ds_array[chan_idx][ds_idx]
		im = ds[im_idx]
		if len(self.shape[chan_idx])==2:
			im = np.expand_dims(im, -1)
		im = im.astype(self.dtype)
		# apply dataset-wise scaling if information is present in attributes
		off = ds.attrs.get('scaling_center', [0])[0]
		factor = ds.attrs.get('scaling_factor', [1])[0]
		if off!=0 or factor!=1:
			im = (im - off)/factor

		# apply group-wise scaling
		off, factor = self._get_scaling(chan_idx, ds_idx)
		if off!=0 or factor!=1:
			im = (im - off) / factor
		return im

	def _get_scaling(self, chan_idx, ds_idx):
		if self.channel_scale==None or self.channel_scale[chan_idx]==None:
			return (0, 1)
		else:
			return self.channel_scale[chan_idx][ds_idx]

	def _get_dataset_path(self, channel_idx, ds_idx):
		if channel_idx==0:
			return self.paths[ds_idx]
		else:
			return self.paths[ds_idx].replace(self.channel_keywords[0], self.channel_keywords[channel_idx])

# class util methods
def copy_affine_tranform_parameters(aug_param_source, aug_param_dest):
		aug_param_dest['theta'] = aug_param_source.get('theta', 0)
		aug_param_dest['tx'] = aug_param_source.get('tx', 0)
		aug_param_dest['ty'] = aug_param_source.get('ty', 0)
		aug_param_dest['shear'] = aug_param_source.get('shear', 0)
		aug_param_dest['zx'] = aug_param_source.get('zx', 1)
		aug_param_dest['zy'] = aug_param_source.get('zy', 1)
		aug_param_dest['flip_horizontal'] = aug_param_source.get('flip_horizontal', False)
		aug_param_dest['flip_vertical'] = aug_param_source.get('flip_vertical', 0)
