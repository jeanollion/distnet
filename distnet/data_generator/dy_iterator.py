from dataset_iterator import TrackingIterator
import numpy as np
from scipy.ndimage import center_of_mass, find_objects, maximum_filter
from scipy.ndimage.measurements import mean
from math import copysign
import sys
import itertools
import edt

class DyIterator(TrackingIterator):
    def __init__(self,
        dataset,
        channel_keywords:list=['/raw', '/regionLabels', '/prevRegionLabels'], # channel @1 must be label & @2 previous label
        next:bool = True,
        return_categories:bool = True,
        closed_end:bool = True,
        erase_cut_cell_length:int = 10,
        aug_remove_prob:float = 0.03,
        aug_frame_subsampling = 1, # either int: subsampling interval will be drawn uniformly in [1,aug_frame_subsampling] or callable that generate an subsampling interval (int)
        **kwargs):
        if len(channel_keywords)!=3:
            raise ValueError('keyword should contain 3 elements in this order: grayscale input images, object labels, object previous labels')

        self.return_categories=return_categories
        self.closed_end=closed_end
        self.erase_cut_cell_length=erase_cut_cell_length
        self.aug_frame_subsampling=aug_frame_subsampling
        super().__init__(dataset=dataset,
                    channel_keywords=channel_keywords,
                    input_channels=[0],
                    output_channels=[1, 2],
                    channels_prev=[True]*3,
                    channels_next=[next]*3,
                    mask_channels=[1, 2],
                    aug_remove_prob=aug_remove_prob,
                    aug_all_frames=False,
                    **kwargs)

    def _get_batch_by_channel(self, index_array, perform_augmentation, input_only=False):
        if self.aug_frame_subsampling!=1 and self.aug_frame_subsampling is not None:
            if callable(self.aug_frame_subsampling):
                self.n_frames = self.aug_frame_subsampling()
            else:
                self.n_frames=np.random.randint(self.aug_frame_subsampling)+1
        return super()._get_batch_by_channel(index_array, perform_augmentation, input_only)

    def _get_input_batch(self, batch_by_channel, ref_chan_idx, aug_param_array):
        input = super()._get_input_batch(batch_by_channel, ref_chan_idx, aug_param_array)
        return_next = self.channels_next[1]
        n_frames = (input.shape[-1]-1)//2 if return_next else input.shape[-1]-1
        if n_frames>1:
            sel = [0, n_frames, -1] if return_next else [0, -1]
            return input[..., sel] # only return
        else:
            return input

    def _get_output_batch(self, batch_by_channel, ref_chan_idx, aug_param_array):
        # dy is computed and returned instead of labels & prevLabels
        labelIms = batch_by_channel[1]
        prevlabelIms = batch_by_channel[2]
        return_next = self.channels_next[1]
        n_frames = (labelIms.shape[-1]-1)//2 if return_next else labelIms.shape[-1]-1
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
            self._erase_small_objects_at_border(labelIms[i,...,n_frames], i, mask_to_erase_cur, mask_to_erase_chan_cur, batch_by_channel)
            # prev timepoint
            self._erase_small_objects_at_border(labelIms[i,...,0], i, mask_to_erase_prev, mask_to_erase_chan_prev, batch_by_channel)
            if return_next:
                self._erase_small_objects_at_border(labelIms[i,...,-1], i, mask_to_erase_next, mask_to_erase_chan_next, batch_by_channel)

        dyIm = np.zeros(labelIms.shape[:-1]+(2 if return_next else 1,), dtype=self.dtype)
        if self.return_categories:
            categories = np.zeros(labelIms.shape[:-1]+(1,), dtype=self.dtype)
            if return_next:
                categories_next = np.zeros(labelIms.shape[:-1]+(1,), dtype=self.dtype)

        for i in range(labelIms.shape[0]):
            prev_start = n_frames - aug_param_array[i][ref_chan_idx].get('oob_inc', n_frames+1) + 1
            _compute_dy(labelIms[i,...,:n_frames+1], prevlabelIms[i,...,prev_start:n_frames+1] if prev_start<n_frames+1 else None, dyIm[i,...,0], categories[i,...,0] if self.return_categories else None)
            if return_next:
                _compute_dy(labelIms[i,...,n_frames:], prevlabelIms[i,...,n_frames:], dyIm[i,...,1], categories_next[i,...,0] if self.return_categories else None)

        other_output_channels = [chan_idx for chan_idx in self.output_channels if chan_idx!=1 and chan_idx!=2]
        all_channels = [batch_by_channel[chan_idx] for chan_idx in other_output_channels]
        all_channels.insert(0, dyIm)
        if self.return_categories:
            all_channels.insert(1, categories)
            if return_next:
                all_channels.insert(2, categories_next)

        edm_c = 3 if return_next else 2
        chan_map = {0:0, 1:n_frames, 2:-1}
        edm = np.zeros(shape = labelIms.shape[:-1]+(edm_c,), dtype=np.float32)
        y_up = 1 if self.closed_end else 0
        for b,c in itertools.product(range(edm.shape[0]), range(edm.shape[-1])):
            # padding along x axis + black_border = False to take into account that cells can go out from upper / lower borders
            edm[b,...,c] = edt.edt(np.pad(labelIms[b,...,chan_map[c]], pad_width=((y_up, 0),(1, 1)), mode='constant', constant_values=0), black_border=False)[y_up:,1:-1]
        all_channels.append(edm)
        return all_channels

    def _erase_small_objects_at_border(self, labelImage, batch_idx, channel_idxs, channel_idxs_chan, batch_by_channel):
        labels_to_erase = _get_small_objects_at_boder_to_erase(labelImage, self.erase_cut_cell_length, self.closed_end)
        if len(labels_to_erase)>0:
            # erase in all mask image then in label image
            slice = labelImage == labels_to_erase
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
def _get_prev_lab(prevlabelIm, labelIm, label, center):
    prev_lab = int(prevlabelIm[int(round(center[0])), int(round(center[1]))])
    if prev_lab==0: # check that mean value is also 0 in the whole, in case center in not included in object
        prev_lab = int(round(mean(prevlabelIm, labelIm, label)))
    return prev_lab

def _get_labels_and_centers(labelIm):
    labels = np.unique(labelIm)
    if len(labels)==0:
        return [],[]
    labels = [int(round(l)) for l in labels if l!=0]
    centers = center_of_mass(labelIm, labelIm, labels)
    return dict(zip(labels, centers))

def _compute_dy(labelIm, prevlabelIm, dyIm, categories=None):
    labels_map_centers = [_get_labels_and_centers(labelIm[...,c]) for c in range(labelIm.shape[-1])]

    if len(labels_map_centers[-1])==0:
        return np.zeros(labelIm.shape[:-1], dtype=labelIm.dtype)
    if prevlabelIm is None: # previous (augmented) image is current image
        labels_map_prev = dict(zip(labels_map_centers[-1].keys(), labels_map_centers[-1].keys()))
    else:
        labels_map_prev = []
        for c in range(1, labelIm.shape[-1]):
            prev_c = c - labelIm.shape[-1]
            if -prev_c<=prevlabelIm.shape[-1]:
                labels_map_prev.append( {label:_get_prev_lab(prevlabelIm[...,prev_c], labelIm[...,c], label, center) for label, center in labels_map_centers[c].items()} )
        if len(labels_map_prev) == 1:
            labels_map_prev = labels_map_prev[0]
        elif len(labels_map_prev)==0: # no previous labels
            labels_map_prev = dict(zip(labels_map_centers[-1].keys(), labels_map_centers[-1].keys()))
        else: # iterate through lineage
            labels_map_prev_ = labels_map_prev[-1]
            for c in range(len(labels_map_prev)-2, -1, -1):
                labels_map_prev__ = labels_map_prev[c]
                get_prev = lambda label : labels_map_prev__[label] if label in labels_map_prev__ else 0
                labels_map_prev_ = {label:get_prev(prev) for label,prev in labels_map_prev_.items()}
            labels_map_prev = labels_map_prev_

    curLabelIm = labelIm[...,-1]
    labels_prev = labels_map_centers[0].keys()
    for label, center in labels_map_centers[-1].items():
        label_prev = labels_map_prev[label]
        if label_prev in labels_prev:
            dy = center[0] - labels_map_centers[0][label_prev][0] # axis 0 is y
            if categories is None and abs(dy)<1:
                dy = copysign(1, dy) # min value == 1 / same sign as dy
            dyIm[curLabelIm == label] = dy

    if categories is not None:
        labels_of_prev_counts = dict(zip(*np.unique(list(labels_map_prev.values()), return_counts=True)))
        for label, label_prev in labels_map_prev.items():
            if label_prev>0 and label_prev not in labels_prev: # no previous
                value=3
            elif labels_of_prev_counts.get(label_prev, 0)>1: # division
                value=2
            else: # previous has single next
                value=1
            categories[curLabelIm == label] = value

def has_object_at_y_borders(mask_img):
    return np.any(mask_img[[-1,0], :], 1) # np.flip()
