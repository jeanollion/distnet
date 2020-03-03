from keras_preprocessing.image import ImageDataGenerator
import numpy as np
import scipy.ndimage as ndi
import distnet.utils.pre_processing as pp

class ImageDataGeneratorSwim(ImageDataGenerator):
    def __init__(self, swim_params=None, swim_params_prev=None, gap_index=True, normalize_intensity=False, **kwargs):
        self.counter = 0
        self.normalize_intensity = normalize_intensity
        self.gap_index=gap_index
        if swim_params and swim_params_prev:
            assert len(swim_params) == len(swim_params_prev), "cur and previous swim parameter should have same length"
            self.swim_params=[]
            for i in range(len(swim_params)):
                self.swim_params.append(swim_params[i])
                self.swim_params.append(swim_params_prev[i])
        elif swim_params:
            self.swim_params=[]
            for i in range(len(swim_params)):
                self.swim_params.append(swim_params[i])
                self.swim_params.append([])
        elif swim_params_prev:
            self.swim_params=[]
            for i in range(len(swim_params)):
                self.swim_params.append([])
                self.swim_params.append(swim_params_prev[i])
        else:
            self.swim_params = None
        super().__init__(**kwargs)

    def resetCounter(self):
        self.counter=0

    def adjust_augmentation_param_from_mask(self, params, mask_img):
        if self.swim_params is not None:
            if self.counter>=len(self.swim_params):
                self.counter = 0
            bact_swim = self.swim_params[self.counter]
            self.counter += 1
            if len(bact_swim)>0:
                if self.gap_index: # convert gap index into y
                    space_y = np.invert(np.any(mask_img, 1)).astype(np.int) # 1 where no label along y axis:
                    space_y, n_lab = ndi.label(space_y)
                    space_y = ndi.find_objects(space_y)
                    space_y = [slice_obj[0] for slice_obj in space_y] # only first dim
                    space_y = [slice_obj for slice_obj in space_y]
                    space_y = [(slice_obj.stop + slice_obj.start)//2 for slice_obj in space_y]
                    bact_swim = [[space_y[space_idx], swim_dist] for [space_idx, swim_dist] in bact_swim]
                # translation will be applied successively -> all following translations are shifted
                for i in range(len(bact_swim)-1):
                    y = bact_swim[i][0]
                    if y>=0 and y<mask_img.shape[0]:
                        dist = bact_swim[i][1] # can be negative
                        for j in range(i+1, len(bact_swim)):
                            y2 = bact_swim[j][0]
                            if dist>0 and y2>y:
                                bact_swim[j][0] += dist
                            elif dist<0 and y2<y:
                                bact_swim[j][0] -= dist
                params["bacteria_swim"] = [[y, dist] for [y, dist] in bact_swim if y>=0 and y<mask_img.shape[0]]
            print(self.count-1, params)

    def apply_transform(self, img, params):
        # swim before geom augmentation with because location must be precisely between 2 bacteria
        if "bacteria_swim" in params:
            for [y, dist] in params["bacteria_swim"]:
                translation_params = {"tx" : - dist}
                if dist>0:
                    img[y:] = super().apply_transform(img[y:], translation_params)
                else:
                    img[:y] = super().apply_transform(img[:y], translation_params)

        # geom augmentation
        img = super().apply_transform(img, params)

        # intensity normalization
        if self.normalize_intensity:
            img = pp.adjust_histogram_range(img)
        return img
