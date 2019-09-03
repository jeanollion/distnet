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
				group_keyword=None,
				image_data_generators=None,
				batch_size=32,
				shuffle=True,
				perform_data_augmentation=True,
				seed=None,
				dtype='float32'):
		if len(channel_keywords)!=3:
			raise ValueError('keyword should have exactly 3 elements: input images, object masks, object displacement')
		super().__init__(h5py_file, channel_keywords, channel_scaling_param, group_keyword, image_data_generators, batch_size, shuffle, perform_data_augmentation, seed)


	def _get_input_batch(self, index_ds, index_array, aug_param_array=None):
		batch = self._get_batches_of_transformed_samples_by_channel(index_ds, index_array, 0, True,  aug_param_array, perform_augmantation=False) # will not populate aug_param_array only if there is an image_data_generator
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
				if aug_param_array[i].get('tx', 0)!=0:
					tx = aug_param_array[i]['tx']
					has_object_up, has_object_down = self._has_object_at_y_borders(index_ds[i], index_array[i]) # up & down as in the displayed image
					if has_object_down and has_object_up:
						aug_param_array[i]['tx']=0
					elif (has_object_up and not has_object_down and tx<0) or (has_object_down and not has_object_up and tx>0):
						aug_param_array[i]['tx'] = -tx
				im = image_data_generator.apply_transform(im, aug_param_array[i])
				batch[i] = image_data_generator.standardize(im)
		batch_prev = self._get_prev_images(index_ds, index_array, batch, aug_param_array) #aslo performs data augmentation for prev + sets dy shift in parameter array
		return np.concatenate((batch_prev, batch), axis=-1)

	def _has_object_at_y_borders(self, ds_idx, im_idx):
		ds = self.ds_array[1][ds_idx]
		off = ds.attrs.get('scaling_center', [0])[0] # supposes there are no other scaling for edm
		return np.any(ds[im_idx, [-1,0], :] - off, 1) # np.flip()

	def _get_prev_images(self, index_ds, index_array, current_batch, aug_param_array=None, aug_remove_prev_prob=0.25):
		im_shape = self.shape[0]
		channel = () if len(im_shape)==3 else (1,)
		prev = np.zeros((len(index_array),) + im_shape + channel, dtype=self.dtype)
		image_data_generator = self.image_data_generators[0] if self.perform_data_augmentation and self.image_data_generators!=None else None
		if image_data_generator is not None and aug_param_array is None:
			raise ValueError("aug_param_array argument should not be none when data augmentation is performed")

		for i, (ds_idx, im_idx) in enumerate(zip(index_ds, index_array)):
			prev_lab = get_prev_label(self.labels[ds_idx][im_idx])
			if im_idx==0 or prev_lab!=self.labels[ds_idx][im_idx-1]: # current (augmented) image is set as prev image + signal in order to erase displacement map in further steps
				prev[i] = current_batch[i]
				if aug_param_array is not None:
					aug_param_array[i]['no_prev'] = True
			elif self.perform_data_augmentation and random.random() < aug_remove_prev_prob: # current image is set as prev image + signal in order to set constant displacement map in further steps
				prev[i] = self._read_image(0, ds_idx, im_idx)
				if aug_param_array is not None:
					aug_param_array[i]['no_prev_aug'] = True
			else:
				prev[i] = im = self._read_image(0, ds_idx, im_idx-1)

			# perform data augmentation on previous (not in the case there are not previous because there is no displacement) + set dy shift in parameter array
			if image_data_generator!=None and not aug_param_array[i].get('no_prev', False):
				params = image_data_generator.get_random_transform(prev[i].shape)
				params['flip_vertical'] = aug_param_array[i].get('flip_vertical', False) # flip must be the same
				prev[i] = image_data_generator.apply_transform(prev[i], params)
				prev[i] = image_data_generator.standardize(prev[i])
				aug_param_array[i]['dy_shift'] = params.get('tx', 0) - aug_param_array[i].get('tx', 0)

		return prev

	def _get_output_batch(self, index_ds, index_array, aug_param_array=None):
		edm = self._get_batches_of_transformed_samples_by_channel(index_ds, index_array, 1, False, aug_param_array)
		dis = self._get_batches_of_transformed_samples_by_channel(index_ds, index_array, 2, False, aug_param_array)
		if aug_param_array is not None:
			# modify displacement value according to delta shift between previous and current indicated in parameter array
			# only takes into account shift in y direction and flip. shear and zoom transform are not taken into account thus they should not be too important
			def add_shift(v, dy):
				return v+dy if v!=0 else 0
			def set_cst_shift(v, dy):
				return dy if v!=0 else 0
			vset_cst_shift = np.vectorize(set_cst_shift)
			vadd_shift = np.vectorize(add_shift)

			for i in range(len(index_ds)):
				if aug_param_array[i].get('no_prev', False):
					dis[i].fill(0)
				elif aug_param_array[i].get('no_prev_aug', False): # current was set as previous image due to image augmentation
					dy = aug_param_array[i].get('dy_shift', 0)
					if dy==0:
						dis[i].fill(0)
					else: # set all non-zero to dy
						dis[i] = vset_cst_shift(dis[i], dy)
				else:
					dy = aug_param_array[i].get('dy_shift', 0)
					if dy!=0: # add dy to all non-zero values
						dis[i] = vadd_shift(dis[i], dy)
				if aug_param_array[i].get('flip_vertical', False):
					dis[i] = -dis[i]
		return np.concatenate((edm, dis), axis=-1)

	def train_test_split(self, **options):
		shuffle_test=options.pop('shuffle_test', self.shuffle)
		perform_data_augmentation_test=options.pop('perform_data_augmentation_test', self.perform_data_augmentation)
		seed_test=options.pop('seed_test', self.seed)
		train_idx, test_idx = train_test_split(self.allowed_indexes, **options)
		train_iterator = H5DisplacementIterator(h5py_file=self.h5py_file,
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
		test_iterator = H5DisplacementIterator(h5py_file=self.h5py_file,
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
def get_prev_label(label):
	frame = int(label[-5:])
	if frame>0:
		frame=frame-1
		frame=str(frame).zfill(5)
		return label[:-5]+frame
	return None
