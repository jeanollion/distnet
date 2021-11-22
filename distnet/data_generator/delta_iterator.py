from dataset_iterator import TrackingIterator
from random import choice
from scipy.ndimage.measurements import center_of_mass
import numpy as np

class DeltaIterator(TrackingIterator):
    def __init__(self,
        dataset,
        channel_keywords=['/raw', '/regionLabels', '/prevRegionLabels'], # channel @1 must be label & @2 previous label
        **kwargs):
        assert len(channel_keywords)==3, "Channels must be: raw, regions labels and preious region labels"
        super().__init__(dataset=dataset,
                         channel_keywords=channel_keywords,
                         input_channels=[0, 1],
                         output_channels=[2],
                         channels_prev=[True, True, False],
                         channels_next=[False, False, False],
                         mask_channels=[1, 2],
                         **kwargs)

    def _get_input_batch(self, batch_by_channel, ref_chan_idx, aug_param_array):
        # current frame: remove all cells but no_next
        # previous cells
        rawIms = batch_by_channel[0] # nothing to do
        labels = batch_by_channel[1] # channel 0 -> choose one label and mark it in data_augmentation, channel 1 : binarize
        return_labels = np.copy(labels)
        for i in range(labels.shape[0]):
            all_labels = list(np.unique(labels[i,...,0]))
            all_labels.remove(0)
            if len(all_labels)==0:
                return_labels[i,...,1] = 0
                aug_param_array[i][1]["label"] = 0
            else:
                label = choice(all_labels)
                return_labels[i,...,0][return_labels[i,...,0] != label] = 0 # erase all other labels
                aug_param_array[i][1]["label"] = label
        return_labels[return_labels>0] = 1 # next : binarize all regions
        return [rawIms, return_labels]

    def _get_output_batch(self, batch_by_channel, ref_chan_idx, aug_param_array):
        labelIms = batch_by_channel[1][...,1]
        prevLabelIms = batch_by_channel[2][...,0]
        # return 1 channel: 0 = background, 1 = next 2 = next (2nd daughter)
        target = np.zeros(shape=labelIms.shape+(1,))
        for i in range(target.shape[0]):
            labelIm = labelIms[i]
            prevLabelIm = prevLabelIms[i]
            label = aug_param_array[i][1]["label"]
            if label>0:
                labels = np.unique(labelIm[prevLabelIm == label])
                if len(labels)==1:
                    target[i,...,0][labelIm == labels[0]] = 1
                elif len(labels)>1: # division: set the uppermost cell as 1 and the others as 2
                    # find the upper most cell
                    centers = center_of_mass(labelIm, labelIm, index=labels)
                    centers = [c[0] for c in centers]
                    idx_min = centers.index(min(centers))
                    label1 = labels[idx_min]
                    label2 = [l for l in labels if l!=label1]
                    # label the cells
                    target[i,...,0][labelIm== label1] = 1
                    target[i,...,0][labelIm== label2] = 2
        return target
