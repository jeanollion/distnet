from dlutils import TrackingIterator
import numpy as np
from scipy.ndimage import center_of_mass, find_objects, maximum_filter
from scipy.ndimage.measurements import mean
from math import copysign
from dlutils.image_data_generator_mm import has_object_at_y_borders

class DyIterator(TrackingIterator):
	def __init__(self,
		h5py_file_path,
		channel_keywords=['/raw', '/regionLabels', '/prevRegionLabels', '/edm'], # channel @1 must be label & @2 previous label
		input_channels=[0],
		output_channels = [1, 2, 3],
		channels_prev=[True, True, False, True],
		channels_next=[False, False, False, False],
		return_categories = False,
		mask_channels=[1, 2, 3],
		output_multiplicity = 1,
		closed_end = True,
		erase_cut_cell_length = 30,
		channel_scaling_param=None, #[{'level':1, 'qmin':5, 'qmax':95}],
		group_keyword=None,
		image_data_generators=None,
		batch_size=32,
		shuffle=True,
		perform_data_augmentation=True,
		seed=None,
		dtype='float32'):
		if len(channel_keywords)<3:
			raise ValueError('keyword should have at least 3 elements in this order: grayscale input images, object labels, object previous labels')
		assert 1 in output_channels
		assert 2 in output_channels
		assert channels_prev[1]
		assert not channels_prev[2]
		assert channels_next[1] == channels_next[2]
		self.return_categories=return_categories
		self.closed_end=closed_end
		self.erase_cut_cell_length=erase_cut_cell_length
		super().__init__(h5py_file_path, channel_keywords, input_channels, output_channels, channels_prev, channels_next, mask_channels, output_multiplicity, channel_scaling_param, group_keyword, image_data_generators, batch_size, shuffle, perform_data_augmentation, seed)

	def _get_output_batch(self, batch_by_channel, ref_chan_idx, aug_param_array):
		# dy is computed and returned instead of labels & prevLabels
		labelIms = batch_by_channel[1]
		prevlabelIms = batch_by_channel[2]
		return_next = self.channels_next[1]

		# remove small objects
		mask_to_erase_cur = [chan_idx for chan_idx in self.mask_channels if chan_idx!=1 and chan_idx in batch_by_channel]
		mask_to_erase_chan_cur = [1 if self.channels_prev[chan_idx] else 0 for chan_idx in mask_to_erase_cur]
		mask_to_erase_prev = [chan_idx for chan_idx in mask_to_erase_cur if self.channels_prev[chan_idx]]
		mask_to_erase_chan_prev = [0] * len(mask_to_erase_prev)
		if return_next:
			mask_to_erase_next = [chan_idx for chan_idx in mask_to_erase_cur if self.channels_next[chan_idx]]
			mask_to_erase_chan_next = [2 if self.channels_prev[chan_idx] else 1 for chan_idx in mask_to_erase_next]

		for i in range(labelIms.shape[0]):
			# cur timepoint
			self._erase_small_objects_at_border(labelIms[i,...,1], i, mask_to_erase_cur, mask_to_erase_chan_cur, batch_by_channel)
			# prev timepoint
			self._erase_small_objects_at_border(labelIms[i,...,0], i, mask_to_erase_prev, mask_to_erase_chan_prev, batch_by_channel)
			if return_next:
				self._erase_small_objects_at_border(labelIms[i,...,2], i, mask_to_erase_next, mask_to_erase_chan_next, batch_by_channel)

		dyIm = np.zeros(labelIms.shape[:-1]+(2 if return_next else 1,), dtype=self.dtype)
		if self.return_categories:
			categories = np.zeros(labelIms.shape[:-1]+(1,), dtype=self.dtype)
			if return_next:
				categories_next = np.zeros(labelIms.shape[:-1]+(1,), dtype=self.dtype)
		for i in range(labelIms.shape[0]):
			if aug_param_array is not None and (aug_param_array[i][ref_chan_idx].get("no_prev", False)):
				prevLabelIm = None
			else:
				prevLabelIm = prevlabelIms[i,...,0]
			_compute_dy(labelIms[i,...,1], labelIms[i,...,0], prevLabelIm, dyIm[i,...,0], categories[i,...,0] if self.return_categories else None)
			if return_next:
				if aug_param_array is not None and (aug_param_array[i][ref_chan_idx].get("no_next", False)):
					prevLabelIm = None
				else:
					prevLabelIm = prevlabelIms[i,...,1]
				_compute_dy(labelIms[i,...,2], labelIms[i,...,1], prevLabelIm, dyIm[i,...,1], categories_next[i,...,0] if self.return_categories else None)

		other_output_channels = [chan_idx for chan_idx in self.output_channels if chan_idx!=1 and chan_idx!=2]

		all_channels = [batch_by_channel[chan_idx] for chan_idx in other_output_channels]
		all_channels.insert(0, dyIm)
		if self.return_categories:
			all_channels.insert(1, categories)
			if return_next:
				all_channels.insert(2, categories_next)
		return all_channels

	def _erase_small_objects_at_border(self, labelImage, batch_idx, channel_idxs, channel_idxs_chan, batch_by_channel):
		labels_to_erase = _get_small_objects_at_boder_to_erase(labelImage, self.erase_cut_cell_length, self.closed_end)
		if len(labels_to_erase)>0:
			# erase in all mask image then in label image
			# dilate image in case labels have been eroded
			dilated = maximum_filter(labelImage, 5) # size >3 in case of zoom
			dilated[labelImage != 0] = labelImage[labelImage != 0] # make sure other labels are not affected by dilatation
			slice = dilated == labels_to_erase
			for mask_chan_idx, c in zip(channel_idxs, channel_idxs_chan):
				batch_by_channel[mask_chan_idx][batch_idx,...,c][slice]=0
			labelImage[slice] = 0

def _get_small_objects_at_boder_to_erase(labelIm, min_length, closed_end):
	has_object_down, has_object_up = has_object_at_y_borders(labelIm)
	res=set()
	if closed_end: # only consider lower part
		has_object_up = False
	if has_object_up or has_object_down:
		 stop = labelIm.shape[0]
		 objects = find_objects(labelIm.astype(np.int))
		 objects = [o[0] if o is not None else None for o in objects] # keep only first dim # none when missing label
		 for l, o in enumerate(objects):
			 if o is not None:
				 if (not closed_end and o.start==0 and (o.stop - o.start)<min_length) or (o.stop==stop and (o.stop - o.start)<min_length):
					 res.add(l+1)
	return list(res)
# dy computation utils
def _get_prev_lab(labelIm_of_prevCells, labelIm, label, center):
	prev_lab = int(labelIm_of_prevCells[int(round(center[0])), int(round(center[1]))])
	if prev_lab==0: # check that mean value is also 0
		prev_lab = int(round(mean(labelIm_of_prevCells, labelIm, label)))
	return prev_lab

def _get_labels_and_centers(labelIm):
	labels = np.unique(labelIm)
	if len(labels)==0:
		return [],[]
	labels = [int(round(l)) for l in labels if l!=0]
	centers = center_of_mass(labelIm, labelIm, labels)
	return labels, centers

def _compute_dy(labelIm, labelIm_prev, labelIm_of_prevCells, dyIm, categories=None):
	labels, centers = _get_labels_and_centers(labelIm)
	if len(labels)==0:
		return np.zeros(labelIm.shape, dtype=labelIm.dtype)
	labels_prev, centers_prev = _get_labels_and_centers(labelIm_prev)
	if labelIm_of_prevCells is None: # previous (augmented) image is current image
		labels_of_prev = labels
	else:
		labels_of_prev = [_get_prev_lab(labelIm_of_prevCells, labelIm, label, center) for label, center in zip(labels, centers)]

	for label, center, label_prev in zip(labels, centers, labels_of_prev):
		if label_prev in labels_prev:
			i_prev = labels_prev.index(label_prev)
			dy = center[0] - centers_prev[i_prev][0] # axis 0 is y
			if categories is None and abs(dy)<1:
				dy = copysign(1, dy) # not 0
			dyIm[labelIm == label] = dy
		#else:
			#dyIm[labelIm == label] = 0
			#sign = 1 if center[0] < dyIm.shape[0] / 2 else -1
			#dyIm[dyIm == label] = dyIm.shape[0] * 2 * sign # not found -> out of the image. What value should be set out-of-the-image ? zero ? other channel ?
	if categories is not None:
		labels_of_prev_counts = dict(zip(*np.unique(labels_of_prev, return_counts=True)))
		for label, label_prev in zip(labels, labels_of_prev):
			if label_prev not in labels_prev: # no previous
				value=3
			elif labels_of_prev_counts.get(label_prev, 0)>1: # division
				value=2
			else: # previous has single next
				value=1
			categories[labelIm == label] = value