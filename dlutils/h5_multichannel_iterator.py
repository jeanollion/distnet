import numpy as np
from dlutils import IndexArrayIterator
from .utils import get_datasets_paths, get_datasets_by_path, get_parent_path
from sklearn.model_selection import train_test_split
import h5py
from math import ceil
import time
class H5MultiChannelIterator(IndexArrayIterator):
	def __init__(self,
				h5py_file,
				channel_keywords=['/raw'],
				channel_scaling_param=[{'level':1, 'qmin':5, 'qmax':95}],
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
		self.channel_scaling_param = channel_scaling_param
		self.dtype = dtype
		self.perform_data_augmentation=perform_data_augmentation
		if image_data_generators!=None and len(channel_keywords)!=len(image_data_generators):
			raise ValueError('image_data_generators argument should be either None or an array of same length as channel_keywords')
		self.image_data_generators=image_data_generators

		# get all dataset paths
		self.paths = get_datasets_paths(self.h5py_file, channel_keywords[0], self.group_keyword)
		if (len(self.paths)==0):
			raise ValueError('No datasets found ending by {} {}'.format(channel_keywords[0], "and containing {}".format(group_keyword) if group_keyword!=None else "" ))
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

		# labels (optional ? )
		self.labels = get_datasets_by_path(self.h5py_file, [path.replace(self.channel_keywords[0], '/labels') for path in self.paths])
		for i, ds in enumerate(self.labels):
			 self.labels[i] = np.char.asarray(ds[()].astype('unicode')) # todo: check if necessary to convert to char array ? unicode is necessary
		if len(self.labels)!=len(self.ds_array[0]):
			raise ValueError('Invalid input file: number of label array differ from dataset number')
		if any(len(self.labels[i].shape)==0 or self.labels[i].shape[0]!=self.ds_array[0][i].shape[0] for i in range(len(self.labels))):
			raise ValueError('Invalid input file: at least one dataset has element numbers that differ from corresponding label array')
		# set scaling information for each dataset
		self.channel_scaling = [None]*len(channel_keywords)
		if self.channel_scaling_param!=None:
			percentile_x = np.arange(0, 101)
			for c, scaling_info in enumerate(self.channel_scaling_param):
				if scaling_info!=None:
					self.channel_scaling[c]=[None]*len(self.paths)
					for ds_idx, path in enumerate(self.paths):
						group = get_parent_path(path)
						for i in range(scaling_info.get('level', 1)):
							group = get_parent_path(group)
							if group==None:
								raise ValueError("scaling group level too high for channel {}({}) group path: {}".format(c, channel_keywords[c]), _get_parent_path(path))
						# percentiles are located in attributes of group
						if not self.h5py_file[group].attrs.__contains__('raw_percentiles'):
							raise ValueError("No percentile array found in group {} for channel: {}({})".format(group, c, channel_keywords[c]))
						percentiles = self.h5py_file[group].attrs.get(channel_keywords[c].replace('/', '')+'_percentiles')
						# get IQR and median
						min, med, max = np.interp([scaling_info.get('qmin', 5), 50, scaling_info.get('qmax', 95)], percentile_x, percentiles)
						self.channel_scaling[c][ds_idx] = [med, max-min]

		super().__init__(indexes[-1], batch_size, shuffle, seed)

	def train_test_split(self, **options):
		"""Split this iterator in two distinct iterators

		Parameters
		----------

		**options : dictionary
		    options passed to train_test_split method of scikit-learn package
			this dictionary can also contain 3 arguments passed to the constructor of the test iterator. if absent, values of the current instance will be passed to the constructor.

			suffle_test : Boolean
			    whether indexes should be shuffled in test iterator
			perform_data_augmentation_test : Boolean
			    wether data augmentation should be performed by the test iterator
			seed_test : Integer
			    seed for test iterator

		Returns
		-------
		tuple of train and test iterators of same type as instance, that access two distinct partitions of the whole dataset.
			train iterator has the same parameters as current instance
			test iterator has the same parameters as current instance except those defined in the argument of this method
		"""

		raise NotImplementedError

	def _get_ds_idx(self, index_array):
		ds_idx = np.searchsorted(self.ds_len, index_array, side='right')
		index_array -= self.ds_off[ds_idx] # remove ds offset to each index
		return ds_idx

	def _get_batches_of_transformed_samples(self, index_array):
		"""Gets a batch of transformed samples.
		# Arguments
			index_array: Array of sample indices to include in batch.
		# Returns
			A batch of transformed samples (tuple of input and output if output_keyword is specified).
		"""
		ds_idx = self._get_ds_idx(index_array)

		if len(self.channel_keywords)==1:
			return self._get_input_batch(ds_idx, index_array)
		elif len(self.channel_keywords)==2 and self.channel_keywords[0]==self.channel_keywords[1]:
			batch = self._get_input_batch(ds_idx, index_array)
			return (batch, batch)
		else:
			aug_param_array = [None]*len(index_array)
			return (self._get_input_batch(ds_idx, index_array, aug_param_array=aug_param_array), self._get_output_batch(ds_idx, index_array, aug_param_array=aug_param_array))

	def _get_input_batch(self, index_ds, index_array, aug_param_array=None):
		"""Generate a batch of input images

		Parameters
		----------
		index_ds : array of integer
		    dataset index for each image
		index_array : array of integer
		    image index within dataset
		aug_param_array : dictionary
		    parameters generated by ImageDataGenerator. Affine transformation parameters are generated for input batch and shared with output batch so that same affine transform are applied to output batch

		Returns
		-------
		numpy array
		    batch of input images

		"""
		raise NotImplementedError

	def _get_output_batch(self, index_ds, index_array, aug_param_array=None):
		"""Generate a batch of output images

		Parameters
		----------
		index_ds : array of integer
		    dataset index for each image
		index_array : array of integer
		    image index within dataset
		aug_param_array : dictionary
		    parameters generated by the ImageDataGenerator of the input channel. Affine transformation parameters are generated for input batch and shared with output batch so that same affine transform are applied to output batch

		Returns
		-------
		numpy array
		    batch of input images

		"""
		raise NotImplementedError

	def _get_batches_of_transformed_samples_by_channel(self, index_ds, index_array, chan_idx, is_input, aug_param_array=None, perform_augmantation=True):
		"""Generate a batch of transformed sample for a given channel

		Parameters
		----------
		index_ds : type
		    dataset index for each image
		index_array : type
		    image index within dataset
		chan_idx : type
		    index of the channel
		is_input : type
		    wether the channel corresponds to the input channel.
			If True, aug_param_array will be populated by this method if there is an ImageDataGenerator for this channel
			If False, affine transformation parameters contained in aug_param_array will be passed to the ImageDataGenerator
		aug_param_array : type
		    parameters generated by the ImageDataGenerator of the input channel.
			Affine transformation parameters are generated for input batch and shared with output batch so that same affine transform are applied to output batch

		Returns
		-------
		type
		    batch of image for the channel of index chan_idx

		"""
		im_shape = self.shape[chan_idx]
		image_data_generator = self.image_data_generators[chan_idx] if self.perform_data_augmentation and perform_augmantation and self.image_data_generators!=None else None
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
				im = image_data_generator.standardize(im)
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
		if self.channel_scaling==None or self.channel_scaling[chan_idx]==None:
			return (0, 1)
		else:
			return self.channel_scaling[chan_idx][ds_idx]

	def _get_dataset_path(self, channel_idx, ds_idx):
		if channel_idx==0:
			return self.paths[ds_idx]
		else:
			return self.paths[ds_idx].replace(self.channel_keywords[0], self.channel_keywords[channel_idx])

	def predict(self, output_file_path, model, output_keys, write_every_n_batches = 100, output_shape = None, apply_to_prediction=None, **create_dataset_options):
		of = h5py.File(output_file_path, 'a')
		if output_shape is None:
			output_shape = self.shape[0]
		self.batch_index=0
		self.perform_data_augmentation=False
		self.shuffle=False
		self._set_index_array()

		if np.any(self.index_array[1:] < self.index_array[:-1]):
			raise ValueError('Index array should be monotonically increasing')

		buffer = [np.zeros(shape = (write_every_n_batches*self.batch_size,)+output_shape, dtype=self.dtype) for c in output_keys]

		for ds_i, ds_i_i, ds_i_len in zip(*np.unique(self._get_ds_idx(self.index_array), return_index=True, return_counts=True)):
			self._ensure_dataset(of, output_shape, output_keys, ds_i, **create_dataset_options)
			paths = [self.paths[ds_i].replace(self.channel_keywords[0], output_key) for output_key in output_keys]
			index_arrays = np.array_split(self.index_array[ds_i_i:(ds_i_i+ds_i_len)], ceil(ds_i_len/self.batch_size))
			print("predictions for dataset:", self.paths[ds_i])
			unsaved_batches = 0
			buffer_idx = 0
			current_indices=[]
			start_pred = time.time()
			for i, index_array in enumerate(index_arrays):
				input = self._get_input_batch([ds_i]*len(index_array), index_array)
				cur_pred = model.predict(input)
				#cur_pred= input
				if apply_to_prediction is not None:
					cur_pred = apply_to_prediction(cur_pred)
				if cur_pred.shape[-1]!=len(output_keys):
					raise ValueError('prediction should have as many channels as output_keys argument')
				if cur_pred.shape[1:-1]!=output_shape:
					raise ValueError("prediction shape differs from output shape")

				for c in range(len(output_keys)):
					buffer[c][buffer_idx:(buffer_idx+input.shape[0])] = cur_pred[..., c]

				buffer_idx+=input.shape[0]
				unsaved_batches +=1
				current_indices.append(index_array)
				if unsaved_batches==write_every_n_batches or i==len(index_arrays)-1:
					start_save = time.time()
					idx_o = list(np.concatenate(current_indices))
					for c in range(len(output_keys)):
						of[paths[c]].write_direct(buffer[c][0:buffer_idx], dest_sel=idx_o)
					end_save = time.time()
					print("#{} batches ({} images) computed in {}s and saved in {}s".format(unsaved_batches, buffer_idx, start_save-start_pred, end_save-start_save))

					unsaved_batches=0
					buffer_idx=0
					current_indices = []
					start_pred = time.time()
		of.close()

	def _ensure_dataset(self, output_file, output_shape, output_keys, ds_i, **create_dataset_options):
		label_path = self.paths[ds_i].replace(self.channel_keywords[0], '/labels')
		if label_path not in output_file:
			output_file.create_dataset(label_path, data = np.asarray(self.labels[ds_i], dtype=np.string_))
		dim_path = self.paths[ds_i].replace(self.channel_keywords[0], '/originalDimensions')
		if dim_path not in output_file and dim_path in self.h5py_file:
			output_file.create_dataset(dim_path, data=self.h5py_file[dim_path])
		for output_key in output_keys:
			ds_path = self.paths[ds_i].replace(self.channel_keywords[0], output_key)
			if ds_path not in output_file:
				output_file.create_dataset(ds_path, (self.ds_array[0][ds_i].shape[0],)+output_shape, dtype=self.dtype, **create_dataset_options) #, compression="gzip" # no compression for compatibility with java driver

	def _has_object_at_y_borders(self, mask_channel_idx, ds_idx, im_idx):
		ds = self.ds_array[mask_channel_idx][ds_idx]
		off = ds.attrs.get('scaling_center', [0])[0] # supposes there are no other scaling for edm
		return np.any(ds[im_idx, [-1,0], :] - off, 1) # np.flip()

	def _forbid_transformations_if_object_touching_borders(self, aug_param, mask_channel_idx, ds_idx, img_idx):
		tx = aug_param.get('tx', 0)
		zx = aug_param.get('zx', 0)
		if tx!=0 or zx<0:
			has_object_up, has_object_down = self._has_object_at_y_borders(mask_channel_idx, ds_idx, img_idx) # up & down as in the displayed image
			if has_object_down and has_object_up:
				aug_param['tx']=0
			elif (has_object_up and not has_object_down and tx<0) or (has_object_down and not has_object_up and tx>0):
				aug_param['tx'] = -tx
			if (has_object_up or has_object_down) and zx<0:
				aug_param['zx'] = 0

# basic implementations
class H5Iterator(H5MultiChannelIterator):
	def __init__(self,
				h5py_file,
				channel_keywords=['/raw'],
				channel_scaling_param=[{'level':1, 'qmin':5, 'qmax':95}],
				group_keyword=None,
				image_data_generators=None,
				batch_size=32,
				shuffle=True,
				perform_data_augmentation=True,
				seed=None,
				dtype='float32'):
		super().__init__(h5py_file, channel_keywords, channel_scaling_param, group_keyword, image_data_generators, batch_size, shuffle, perform_data_augmentation, seed)

	def _get_input_batch(self, index_ds, index_array, aug_param_array=None):
		return self._get_batches_of_transformed_samples_by_channel(index_ds, index_array, 0, True,  aug_param_array)

	def _get_output_batch(self, index_ds, index_array, aug_param_array=None):
		return self._get_batches_of_transformed_samples_by_channel(index_ds, index_array, len(self.channel_keywords)-1, False, aug_param_array)

	def train_test_split(self, **options):
		shuffle_test=options.pop('shuffle_test', self.shuffle)
		perform_data_augmentation_test=options.pop('perform_data_augmentation_test', self.perform_data_augmentation)
		seed_test=options.pop('seed_test', self.seed)
		train_idx, test_idx = train_test_split(self.allowed_indexes, **options)
		train_iterator = H5Iterator(h5py_file=self.h5py_file,
									channel_keywords=self.channel_keywords,
									channel_scaling_param=self.channel_scaling_param,
									group_keyword=self.group_keyword,
									image_data_generators=self.image_data_generators,
									batch_size=self.batch_size,
									shuffle=self.shuffle,
									perform_data_augmentation=self.perform_data_augmentation,
									seed=self.seed,
									dtype=self.dtype)
		train_iterator.set_allowed_indexes(train_idx)
		test_iterator = H5Iterator(h5py_file=self.h5py_file,
								channel_keywords=self.channel_keywords,
								channel_scaling_param=self.channel_scaling_param,
								group_keyword=self.group_keyword,
								image_data_generators=self.image_data_generators,
								batch_size=self.batch_size,
								shuffle=shuffle_test,
								perform_data_augmentation=perform_data_augmentation_test,
								seed=seed_test,
								dtype=self.dtype)
		test_iterator.set_allowed_indexes(test_idx)
		return train_iterator, test_iterator

class H5SegmentationIterator(H5MultiChannelIterator):
	def __init__(self,
				h5py_file,
				channel_keywords=['/raw', '/edm'],
				channel_scaling_param=[{'level':1, 'qmin':5, 'qmax':95}],
				group_keyword=None,
				image_data_generators=None,
				batch_size=32,
				shuffle=True,
				perform_data_augmentation=True,
				seed=None,
				dtype='float32'):
		super().__init__(h5py_file, channel_keywords, channel_scaling_param, group_keyword, image_data_generators, batch_size, shuffle, perform_data_augmentation, seed)

	def _get_input_batch(self, index_ds, index_array, aug_param_array=None):
		batch = self._get_batches_of_transformed_samples_by_channel(index_ds, index_array, 0, True,  aug_param_array, perform_augmantation=False) # will not populate aug_param_array even if there is an image_data_generator
		if not self.perform_data_augmentation or self.image_data_generators==None or self.image_data_generators[0]==None: # aug_param_array is used to signal that there is no previous image in order to set displacements to zero
			if aug_param_array  is not None:
				for i in range(len(aug_param_array)):
					aug_param_array[i] = dict()
		else: # perform data augmentation
			if aug_param_array is None:
				raise ValueError("aug_param_array argument should not be none when data augmentation is performed")
			image_data_generator = self.image_data_generators[0]
			for i, im in enumerate(batch):
				aug_param_array[i] = image_data_generator.get_random_transform(im.shape)
				# check that there are no object @ upper or lower border to avoid generating artefacts with translation along y axis
				self._forbid_transformations_if_object_touching_borders(aug_param_array[i], 1, index_ds[i], index_array[i])
				im = image_data_generator.apply_transform(im, aug_param_array[i])
				batch[i] = image_data_generator.standardize(im)
		return batch
	def _get_output_batch(self, index_ds, index_array, aug_param_array=None):
		return self._get_batches_of_transformed_samples_by_channel(index_ds, index_array, 1, False, aug_param_array)




	def train_test_split(self, **options):
		shuffle_test=options.pop('shuffle_test', self.shuffle)
		perform_data_augmentation_test=options.pop('perform_data_augmentation_test', self.perform_data_augmentation)
		seed_test=options.pop('seed_test', self.seed)
		train_idx, test_idx = train_test_split(self.allowed_indexes, **options)
		train_iterator = H5SegmentationIterator(h5py_file=self.h5py_file,
									channel_keywords=self.channel_keywords,
									channel_scaling_param=self.channel_scaling_param,
									group_keyword=self.group_keyword,
									image_data_generators=self.image_data_generators,
									batch_size=self.batch_size,
									shuffle=self.shuffle,
									perform_data_augmentation=self.perform_data_augmentation,
									seed=self.seed,
									dtype=self.dtype)
		train_iterator.set_allowed_indexes(train_idx)
		test_iterator = H5SegmentationIterator(h5py_file=self.h5py_file,
								channel_keywords=self.channel_keywords,
								channel_scaling_param=self.channel_scaling_param,
								group_keyword=self.group_keyword,
								image_data_generators=self.image_data_generators,
								batch_size=self.batch_size,
								shuffle=shuffle_test,
								perform_data_augmentation=perform_data_augmentation_test,
								seed=seed_test,
								dtype=self.dtype)
		test_iterator.set_allowed_indexes(test_idx)
		return train_iterator, test_iterator



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
