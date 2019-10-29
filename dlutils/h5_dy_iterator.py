from dlutils import H5TrackingIterator
import numpy as np
from scipy.ndimage import center_of_mass
from scipy.ndimage.measurements import mean

class H5dyIterator(H5TrackingIterator):
	def __init__(self,
		h5py_file,
		channel_keywords=['/raw', '/label', '/prevLabel', '/edm'], # channel @1 must be label & @2 previous label
		input_channels=[0, 3],
		output_channels=[1],
		input_channels_prev=[True, True],
		input_channels_next=[False, False],
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
		if len(channel_keywords)<3:
			raise ValueError('keyword should have at least 3 elements: input images, object labels, object previous labels')
		if output_channels[0]!=1:
			raise ValueError('first output channel must be labels and will be converted in displacement')
		super().__init__(h5py_file, channel_keywords, input_channels, output_channels, input_channels_prev, input_channels_next, [True], output_channels_next, mask_channel, channel_scaling_param, group_keyword, image_data_generators, batch_size, shuffle, perform_data_augmentation, seed)
		if 2 in output_channels and self._include_prev(2, False):
			raise ValueError('previous frame of channel 2 (previous labels) cannot be returned')

	def _get_output_batch(self, index_ds, index_array, aug_param_array=None):
		# label and prev label
		labelIms = self._get_batches_of_transformed_samples_by_channel(index_ds, index_array, 1, False, aug_param_array, perform_augmentation=True)
		prevlabelIms = self._get_batches_of_transformed_samples_by_channel(index_ds, index_array, 2, False, aug_param_array, perform_augmentation=True)

		dyIm = np.zeros(labelIms.shape[:-1]+(1,), dtype=self.dtype)
		for i in range(labelIms.shape[0]):
			dyIm[i] = compute_dy(labelIms[i,...,1], labelIms[i,...,0], prevlabelIms[i,...,0])

		other_channels = [self._get_batches_of_transformed_samples_by_channel(index_ds, index_array, chan_idx, False, aug_param_array, perform_augmentation=True) for chan_idx in self.output_channels[1:]]
		other_channels.insert(0, dyIm)
		return other_channels

	def _forbid_transformations_if_object_touching_borders(self, aug_param, mask_channel_idx, ds_idx, img_idx):
		#if aug_param.get('zx', 1)!=1: # forbid zoom in this direction to because displacement would need to be re-computed
		#	aug_param['zx'] = 1
		super()._forbid_transformations_if_object_touching_borders(aug_param, mask_channel_idx, ds_idx, img_idx)

	# def _correct_dy_after_augmentation(self, batch, aug_param_array, prev=True):
	# 	no_neigh = 'no_prev' if prev else 'no_next'
	# 	no_neigh_aug = 'no_prev_aug' if prev else 'no_next_aug'
	# 	dy_shift = 'dy_shift_prev' if prev else 'dy_shift_next'
	#
	# 	# modify displacement value according to delta shift between previous and current indicated in parameter array
	# 	# only takes into account shift in y direction and flip. shear and zoom transform are not taken into account thus they should not be too important
	# 	def add_shift(v, dy):
	# 		return v+dy if v!=0 else 0
	# 	def set_cst_shift(v, dy):
	# 		return dy if v!=0 else 0
	# 	vset_cst_shift = np.vectorize(set_cst_shift)
	# 	vadd_shift = np.vectorize(add_shift)
	#
	# 	for i in range(batch.shape[0]):
	# 		dy = aug_param_array[i][0].get(dy_shift, 0)
	# 		if aug_param_array[i][0].get(no_neigh, False): # no previous / next: no displacement information to displacement should be zero
	# 			batch[i].fill(0)
	# 		elif aug_param_array[i][0].get(no_neigh_aug, False): # current was set as previous/next image due to image augmentation: displacement information is present be has to be set to a constant value
	# 			if dy==0:
	# 				batch[i].fill(0)
	# 			else: # set all non-zero to dy
	# 				batch[i] = vset_cst_shift(batch[i], dy)
	# 		elif dy!=0: # add dy to all non-zero values
	# 			batch[i] = vadd_shift(batch[i], dy)
	# 		if aug_param_array[i][0].get('flip_vertical', False):
	# 			batch[i] = -batch[i]

	# def _get_batches_of_transformed_samples_by_channel(self, index_ds, index_array, chan_idx, is_input, aug_param_array=None, perform_augmentation=True):
	# 	batch = super()._get_batches_of_transformed_samples_by_channel(index_ds, index_array, chan_idx, is_input, aug_param_array, perform_augmentation)
	# 	if chan_idx==2:
	# 		next = self._include_next(chan_idx, is_input)
	# 		if not next:
	# 			self._correct_dy_after_augmentation(batch, aug_param_array, True)
	# 		else: # correct separately cur -> prev et next -> prev
	# 			self._correct_dy_after_augmentation(batch[...,0], aug_param_array, True)
	# 			self._correct_dy_after_augmentation(batch[...,1], aug_param_array, False)
	# 	return batch

# class util methods
def get_prev_lab(labelIm_of_prevCells, labelIm, label, center):
	prev_lab = labelIm_of_prevCells[int(round(center[0])), int(round(center[1]))]
	if prev_lab==0: # check that mean value is also 0
		prev_lab = round(mean(labelIm_of_prevCells, labelIm, label))
	return prev_lab

def get_labels_and_centers(labelIm):
	labels = np.unique(labelIm)
	if len(labels)==0:
		return [],[]
	labels = [int(round(l)) for l in labels if l!=0]
	centers = center_of_mass(labelIm, labelIm, labels)
	return labels, centers

def compute_dy(labelIm, labelIm_prev, labelIm_of_prevCells):
	labels, centers = get_labels_and_centers(labelIm)
	if len(labels)==0:
		return np.zeros(labels.shape, dtype=self.dtype)

	dyIm = np.copy(labelIm)
	prevLabs = [get_prev_lab(labelIm_of_prevCells, labelIm, label, centers[i]) for i, label in enumerate(labels)]
	labels_prev, centers_prev = get_labels_and_centers(labelIm_prev)

	for i, label in enumerate(labels):
		if label not in labels_prev:
			dyIm[dyIm == label] = 0
		i_prev = labels_prev.index(label)
		dy = centers[i][1] - centers_prev[i_prev][1]
		dyIm[dyIm == label] = dy
	return dy
