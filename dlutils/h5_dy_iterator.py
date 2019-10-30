from dlutils import H5TrackingIterator
import numpy as np
from scipy.ndimage import center_of_mass
from scipy.ndimage.measurements import mean

class H5dyIterator(H5TrackingIterator):
	def __init__(self,
		h5py_file,
		channel_keywords=['/raw', '/regionLabels', '/prevRegionLabels', '/edm'], # channel @1 must be label & @2 previous label
		input_channels=[0, 3],
		input_channels_prev=[True, True],
		input_channels_next=[False, False],
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
		super().__init__(h5py_file, channel_keywords, input_channels, [1], input_channels_prev, input_channels_next, [True], [False], mask_channel, channel_scaling_param, group_keyword, image_data_generators, batch_size, shuffle, perform_data_augmentation, seed)

	def _get_output_batch(self, index_ds, index_array, aug_param_array=None):
		# dy is computed and returned
		# label and prev label
		labelIms = self._get_batches_of_transformed_samples_by_channel(index_ds, index_array, 1, False, aug_param_array, perform_augmentation=True)
		prevlabelIms = self._get_batches_of_transformed_samples_by_channel(index_ds, index_array, 2, False, aug_param_array, perform_augmentation=True)
		no_neigh_key = "no_prev"
		no_neigh_aug_key = "no_prev_aug"
		dyIm = np.zeros(labelIms.shape[:-1]+(1,), dtype=self.dtype)
		for i in range(labelIms.shape[0]):
			if aug_param_array is not None and (aug_param_array[i][0].get(no_neigh_key, False) or aug_param_array[i][0].get(no_neigh_aug_key, False)):
				prevLabelIm = None
			else:
				prevLabelIm = prevlabelIms[i,...,0]
			dyIm[i][...,0] = compute_dy(labelIms[i,...,1], labelIms[i,...,0], prevLabelIm)

		if len(self.output_channels)>1:
			all_channels = [self._get_batches_of_transformed_samples_by_channel(index_ds, index_array, chan_idx, False, aug_param_array, perform_augmentation=True) for chan_idx in self.output_channels[1:]]
			all_channels.insert(0, dyIm)
			return all_channels
		else:
			return dyIm

# dy computation utils
def get_prev_lab(labelIm_of_prevCells, labelIm, label, center):
	prev_lab = int(labelIm_of_prevCells[int(round(center[0])), int(round(center[1]))])
	if prev_lab==0: # check that mean value is also 0
		prev_lab = int(round(mean(labelIm_of_prevCells, labelIm, label)))
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
		return np.zeros(labelIm.shape, dtype=labelIm.dtype)
	labels_prev, centers_prev = get_labels_and_centers(labelIm_prev)
	if labelIm_of_prevCells is None: # previous image is (augmented) current image
		labels_of_prev = labels
	else:
		labels_of_prev = [get_prev_lab(labelIm_of_prevCells, labelIm, label, center) for label, center in zip(labels, centers)]

	dyIm = np.copy(labelIm)
	for label, center, label_prev in zip(labels, centers, labels_of_prev):
		if label_prev not in labels_prev:
			dyIm[dyIm == label] = dyIm.shape[0] * 2 # not found -> out of the image ? or zero ?
		else:
			i_prev = labels_prev.index(label_prev)
			dy = center[0] - centers_prev[i_prev][0] # axis 0 is y
			if dy==0:
				dy = 0.1 # not 0
			dyIm[dyIm == label] = dy
	return dyIm
