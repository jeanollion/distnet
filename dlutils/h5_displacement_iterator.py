import numpy as np
from dlutils import H5Iterator
from .utils import get_datasets_by_path
import random

class H5DisplacementIterator(H5Iterator):
	def __init__(self,
				h5py_file,
				channel_keywords=['/raw', '/edm', '/dy'],
				channel_scaling=[{'level':1, 'qmin':5, 'qmax':95}],
				group_keyword=None,
				image_data_generators=None,
				batch_size=32,
				shuffle=True,
				perform_data_augmentation=True,
				seed=None,
				dtype='float32'):
		if len(channel_keywords)!=3:
			raise ValueError('keyword should have exactly 3 elements: input images, object masks, object displacement')
		super(H5DisplacementIterator, self).__init__(h5py_file, channel_keywords, channel_scaling, group_keyword, image_data_generators, batch_size, shuffle, perform_data_augmentation, seed)
		self.labels = get_datasets_by_path(self.h5py_file, [path.replace(self.channel_keywords[0], '/labels') for path in self.paths])
		for i, ds in enumerate(self.labels):
			 self.labels[i] = np.char.asarray(ds[()].astype('unicode')) # todo: check if necessary to convert to char array ? unicode is necessary
		if len(self.labels)!=len(self.ds_array[0]):
			raise ValueError('Invalid input file: number of label array differ from dataset number')
		if any(len(self.labels[i].shape)==0 or self.labels[i].shape[0]!=self.ds_array[0][i].shape[0] for i in range(len(self.labels))):
			raise ValueError('Invalid input file: at least one dataset has element numbers that differ from corresponding label array')

	def _get_input_batch(self, index_ds, index_array, aug_param_array):
		if aug_param_array==None:
			raise ValueError("Parameter array cannot be None")
		batch = self._get_batches_of_transformed_samples_by_channel(index_ds, index_array, 0, True,  aug_param_array) # will populate aug_param_array only if there is an image_data_generator
		if not self.perform_data_augmentation or self.image_data_generators==None or self.image_data_generators[0]==None: # even if there is no data augmentation, aug_param_array is used to signal that there is no previous image in order to set displacements to zero
			for i in range(len(aug_param_array)):
				aug_param_array[i] = dict()
		batch_prev = self._get_prev_images(index_ds, index_array, batch, aug_param_array) #aslo performs data augmentation for prev + sets dy shift in parameter array
		return np.concatenate((batch_prev, batch), axis=-1)

	def _get_prev_images(self, index_ds, index_array, current_batch, aug_param_array, aug_remove_prev_prob=0.25):
		im_shape = self.shape[0]
		channel = () if len(im_shape)==3 else (1,)
		prev = np.zeros((len(index_array),) + im_shape + channel, dtype=self.dtype)
		image_data_generator = self.image_data_generators[0] if self.perform_data_augmentation and self.image_data_generators!=None else None
		for i, (ds_idx, im_idx) in enumerate(zip(index_ds, index_array)):
			prev_lab = get_prev_label(self.labels[ds_idx][im_idx])
			if im_idx==0 or prev_lab!=self.labels[ds_idx][im_idx-1]: # current (augmented) image is set as prev image + signal in order to erase displacement map in further steps
				prev[i] = current_batch[i]
				aug_param_array[i]['no_prev'] = True
			elif self.perform_data_augmentation and random.random() < aug_remove_prev_prob: # current image is set as prev image + signal in order to set constant displacement map in further steps
				prev[i] = self._read_image(0, ds_idx, im_idx)
				aug_param_array[i]['no_prev_aug'] = True
			else:
				prev[i] = im = self._read_image(0, ds_idx, im_idx-1)

			# perform data augmentation on previous (not in the case there are not previous because there is no displacement) + set dy shift in parameter array
			if image_data_generator!=None and not aug_param_array[i].get('no_prev', False):
				params = image_data_generator.get_random_transform(prev[i].shape)
				params['flip_vertical'] = aug_param_array[i].get('flip_vertical', False) # flip must be the same
				prev[i] = image_data_generator.apply_transform(prev[i], params)
				prev[i] = image_data_generator.standardize(prev[i])
				aug_param_array[i]['dy_shift'] = aug_param_array[i].get('ty', 0) - params.get('ty', 0)

		return prev

	def _get_output_batch(self, index_ds, index_array, aug_param_array):
		edm = self._get_batches_of_transformed_samples_by_channel(index_ds, index_array, 1, False, aug_param_array)
		dis = self._get_batches_of_transformed_samples_by_channel(index_ds, index_array, 2, False, aug_param_array)
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

# class util methods
def get_prev_label(label):
	frame = int(label[-5:])
	if frame>0:
		frame=frame-1
		frame=str(frame).zfill(5)
		return label[:-5]+frame
	return None
