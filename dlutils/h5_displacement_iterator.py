import numpy as np
from dlutils import H5MultiChannelIterator
from .utils import get_datasets_by_path
import random
from sklearn.model_selection import train_test_split

class H5DisplacementIterator(H5MultiChannelIterator):
	def __init__(self,
				h5py_file,
				channel_keywords=['/raw', '/edm', '/dy'],
				channel_scaling_param=[{'level':1, 'qmin':5, 'qmax':95}],
				include_next=False,
				return_edm_neigh=False,
				group_keyword=None,
				image_data_generators=None,
				batch_size=32,
				shuffle=True,
				perform_data_augmentation=True,
				seed=None,
				dtype='float32'):
		if len(channel_keywords)!=3:
			raise ValueError('keyword should have exactly 3 elements: input images, object masks, object displacement')
		self.include_next=include_next
		self.return_edm_neigh=return_edm_neigh
		super().__init__(h5py_file, channel_keywords, channel_scaling_param, group_keyword, image_data_generators, batch_size, shuffle, perform_data_augmentation, seed)

	def _get_input_batch(self, index_ds, index_array, aug_param_array=None):
		batch = self._get_batches_of_transformed_samples_by_channel(index_ds, index_array, 0, True,  aug_param_array, perform_augmantation=False) # will not populate aug_param_array even if there is an image_data_generator
		aug = self.perform_data_augmentation and self.image_data_generators!=None and self.image_data_generators[0]!=None
		if not aug: # aug_param_array is used to signal that there is no previous image in order to set displacements to zero
			if aug_param_array is not None:
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

		batch_prev = self._get_neighbor_images(index_ds, index_array, batch, True, aug_param_array) #aslo performs data augmentation for prev + sets dy shift in parameter array
		batch_next = self._get_neighbor_images(index_ds, index_array, batch, False, aug_param_array) if self.include_next else None
		if aug: # augmentation if only performed after get prev & next batch so raw image can be copied instead of read them again
			for i, im in enumerate(batch):
				im = image_data_generator.apply_transform(im, aug_param_array[i])
				batch[i] = image_data_generator.standardize(im)

		if self.include_next:
			return np.concatenate((batch_prev, batch, batch_next), axis=-1)
		else:
			return np.concatenate((batch_prev, batch), axis=-1)

	def _get_neighbor_images(self, index_ds, index_array, current_batch, prev, aug_param_array=None, aug_remove_prob=0.2):
		im_shape = self.shape[0]
		channel = () if len(im_shape)==3 else (1,)
		batch = np.zeros((len(index_array),) + im_shape + channel, dtype=self.dtype)
		image_data_generator = self.image_data_generators[0] if self.perform_data_augmentation and self.image_data_generators!=None else None
		if image_data_generator is not None and aug_param_array is None:
			raise ValueError("aug_param_array argument should not be none when data augmentation is performed")
		key = "no_prev" if prev else "no_next"
		key_aug = "no_prev_aug" if prev else "no_next_aug"
		inc = -1 if prev else 1
		for i, (ds_idx, im_idx) in enumerate(zip(index_ds, index_array)):
			neigh_lab = get_neighbor_label(self.labels[ds_idx][im_idx], prev=prev)
			bound_idx = 0 if prev else len(self.labels[ds_idx])-1
			if im_idx==bound_idx or neigh_lab!=self.labels[ds_idx][im_idx+inc]: # current (augmented) image is set as prev/next image + signal in order to erase displacement map in further steps
				batch[i] = np.copy(current_batch[i])
				if aug_param_array is not None:
					aug_param_array[i][key] = True
			elif self.perform_data_augmentation and random.random() < aug_remove_prob: # current image is set as prev/next image + signal in order to set constant displacement map in further steps
				batch[i] = np.copy(current_batch[i])
				if aug_param_array is not None:
					aug_param_array[i][key_aug] = True
			else:
				batch[i] = self._read_image(0, ds_idx, im_idx+inc)

			# perform data augmentation on neighbor image
			if image_data_generator!=None:
				aug_key = "aug_params_prev" if prev else "aug_params_next"
				dy_shift_key = 'dy_shift_prev' if prev else 'dy_shift_next'
				params = image_data_generator.get_random_transform(batch[i].shape)
				params['flip_vertical'] = aug_param_array[i].get('flip_vertical', False) # flip must be the same
				params['zy'] = aug_param_array[i].get('zy', 1) # zoom should be the same so that cell aspect does not changes too much
				params['brightness_'] = aug_param_array[i].get('brightness_', 0)
				params['contrast_'] = aug_param_array[i].get('contrast_', 1)
				
				if aug_param_array[i].get(key, False): # there is no displacement data so shift must be 0
					params['tx']=aug_param_array[i].get('tx', 0)
				elif aug_param_array[i].get(key_aug, False): # displacement image is the current one
					self._forbid_transformations_if_object_touching_borders(params, 1, ds_idx, im_idx)
				else:
					self._forbid_transformations_if_object_touching_borders(params, 1, ds_idx, im_idx+inc)
				batch[i] = image_data_generator.apply_transform(batch[i], params)
				batch[i] = image_data_generator.standardize(batch[i])
				aug_param_array[i][aug_key] = params
				aug_param_array[i][dy_shift_key] = (aug_param_array[i].get('tx', 0) - params.get('tx', 0)) * inc

		return batch

	def _get_batches_of_transformed_samples_by_channel_neighbor(self, index_ds, index_array, chan_idx, is_input, aug_param_array, prev, perform_augmantation=True):
		# define new index array. no prev / no next tag is in the aug_param_array
		inc = -1 if prev else 1
		no_neigh_key = "no_prev" if prev else "no_next"
		no_neigh_aug_key = "no_prev_aug" if prev else "no_next_aug"
		aug_key = "aug_params_prev" if prev else "aug_params_next"
		index_array = np.copy(index_array)
		index_array += inc
		missing_idx = [i for i in range(len(aug_param_array)) if aug_param_array[i].get(no_neigh_key, False) or aug_param_array[i].get(no_neigh_aug_key, False)]
		index_array[missing_idx] -= inc
		aug_param_array = [aug_param_array[i].get(aug_key, aug_param_array[i]) for i in range(len(aug_param_array))]
		return self._get_batches_of_transformed_samples_by_channel(index_ds, index_array, chan_idx, is_input, aug_param_array, perform_augmantation)

	def _get_output_batch(self, index_ds, index_array, aug_param_array=None):
		edm = self._get_batches_of_transformed_samples_by_channel(index_ds, index_array, 1, False, aug_param_array)
		dis = self._get_batches_of_transformed_samples_by_channel(index_ds, index_array, 2, False, aug_param_array)
		if aug_param_array is not None:
			self._correct_dy_after_augmentation(dis, aug_param_array, prev=True)
		if not self.include_next:
			if self.return_edm_neigh:
				edm_prev = self._get_batches_of_transformed_samples_by_channel_neighbor(index_ds, index_array, 1, False, aug_param_array, prev=True)
				return np.concatenate((edm_prev, edm, dis), axis=-1)
			else:
				return np.concatenate((edm, dis), axis=-1)
		else:
			dis_next = self._get_batches_of_transformed_samples_by_channel_neighbor(index_ds, index_array, 2, False, aug_param_array, prev=False)
			self._correct_dy_after_augmentation(dis_next, aug_param_array, prev=False)
			if self.return_edm_neigh:
				edm_prev = self._get_batches_of_transformed_samples_by_channel_neighbor(index_ds, index_array, 1, False, aug_param_array, prev=True)
				edm_next = self._get_batches_of_transformed_samples_by_channel_neighbor(index_ds, index_array, 1, False, aug_param_array, prev=False)
				return np.concatenate((edm_prev, edm, edm_next, dis, dis_next), axis=-1)
			else:
				return np.concatenate((edm, dis, dis_next), axis=-1)

	def _correct_dy_after_augmentation(self, batch, aug_param_array, prev=True):
		no_neigh = 'no_prev' if prev else 'no_next'
		no_neigh_aug = 'no_prev_aug' if prev else 'no_next_aug'
		dy_shift = 'dy_shift_prev' if prev else 'dy_shift_next'

		# modify displacement value according to delta shift between previous and current indicated in parameter array
		# only takes into account shift in y direction and flip. shear and zoom transform are not taken into account thus they should not be too important
		def add_shift(v, dy):
			return v+dy if v!=0 else 0
		def set_cst_shift(v, dy):
			return dy if v!=0 else 0
		vset_cst_shift = np.vectorize(set_cst_shift)
		vadd_shift = np.vectorize(add_shift)

		for i in range(batch.shape[0]):
			dy = aug_param_array[i].get(dy_shift, 0)
			if aug_param_array[i].get(no_neigh, False):
				batch[i].fill(0)
			elif aug_param_array[i].get(no_neigh_aug, False): # current was set as previous/next image due to image augmentation
				if dy==0:
					batch[i].fill(0)
				else: # set all non-zero to dy
					batch[i] = vset_cst_shift(batch[i], dy)
			elif dy!=0: # add dy to all non-zero values
				batch[i] = vadd_shift(batch[i], dy)
			if aug_param_array[i].get('flip_vertical', False):
				batch[i] = -batch[i]

	def _forbid_transformations_if_object_touching_borders(self, aug_param, mask_channel_idx, ds_idx, img_idx):
		if aug_param.get('zx', 1)!=1: # forbid zoom in this direction because
			aug_param['zx'] = 1
		super()._forbid_transformations_if_object_touching_borders(aug_param, mask_channel_idx, ds_idx, img_idx)

	def train_test_split(self, **options):
		shuffle_test=options.pop('shuffle_test', self.shuffle)
		perform_data_augmentation_test=options.pop('perform_data_augmentation_test', self.perform_data_augmentation)
		seed_test=options.pop('seed_test', self.seed)
		train_idx, test_idx = train_test_split(self.allowed_indexes, **options)
		# remove neighboring values that are seen by the network. only in terms of ground truth, ie depends on returned values: self.return_edm_neigh and self.include_next: previous and next frames. only self.include_next: next frame only (displacement)
		if self.return_edm_neigh: # an index visited in train_idx implies the previous one is also seen during training. to avoind that previous index being in test_idx, next indices of test_idx should remove from train_idx
			train_idx = np.setdiff1d(train_idx, self._get_neighbor_indices(test_idx, prev=False))
		if self.include_next: # an index visited in train_idx implies the next one is also seen during training. to avoind that next index being in test_idx, previous indices of test_idx should remove from train_idx
			train_idx = np.setdiff1d(train_idx, self._get_neighbor_indices(test_idx, prev=True))
		# need to exclude indices before / after test_idx from train_idx ?
		train_iterator = H5DisplacementIterator(h5py_file=self.h5py_file,
		                            channel_keywords=self.channel_keywords,
		                            channel_scaling_param=self.channel_scaling_param,
									include_next=self.include_next,
									return_edm_neigh=self.return_edm_neigh,
		                            group_keyword=self.group_keyword,
		                            image_data_generators=self.image_data_generators,
		                            batch_size=self.batch_size,
		                            shuffle=self.shuffle,
		                            perform_data_augmentation=self.perform_data_augmentation,
		                            seed=self.seed,
		                            dtype=self.dtype)
		train_iterator.set_allowed_indexes(train_idx)
		test_iterator = H5DisplacementIterator(h5py_file=self.h5py_file,
		                            channel_keywords=self.channel_keywords,
		                            channel_scaling_param=self.channel_scaling_param,
									include_next=self.include_next,
									return_edm_neigh=self.return_edm_neigh,
		                            group_keyword=self.group_keyword,
		                            image_data_generators=self.image_data_generators,
		                            batch_size=self.batch_size,
		                            shuffle=shuffle_test,
									perform_data_augmentation=perform_data_augmentation_test,
									seed=seed_test,
		                            dtype=self.dtype)
		test_iterator.set_allowed_indexes(test_idx)
		return train_iterator, test_iterator

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
