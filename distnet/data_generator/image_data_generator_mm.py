from keras_preprocessing.image import ImageDataGenerator
#from tensorflow.keras.preprocessing.image import ImageDataGenerator # this version doesn't have interpolation_order
import numpy as np
from math import tan, atan, pi, copysign
import distnet.utils.pre_processing as pp
from random import getrandbits, uniform, choice
import copy
import scipy.ndimage as ndi

class ImageDataGeneratorMM(ImageDataGenerator):
    def __init__(self, rotate90=False, width_zoom_range=0., height_zoom_range=0., max_zoom_aspectratio=1.5, min_zoom_aspectratio=0., perform_illumination_augmentation = True, gaussian_blur_range=[1, 2], noise_intensity = 0.1, min_histogram_range=0.1, min_histogram_to_zero=False, histogram_normalization_center=None, histogram_normalization_scale=None, histogram_voodoo_n_points=5, histogram_voodoo_intensity=0.5, illumination_voodoo_n_points=5, illumination_voodoo_intensity=0.6, bacteria_swim_distance=50, bacteria_swim_min_gap=3, closed_end=True, **kwargs):
        if width_zoom_range is None:
            width_zoom_range=0
        if height_zoom_range is None:
            height_zoom_range=0
        if gaussian_blur_range is None:
            gaussian_blur_range=0
        if width_zoom_range!=0. or height_zoom_range!=0.:
            kwargs["zoom_range"] = 0.
            if np.isscalar(width_zoom_range):
                self.width_zoom_range = [1 - width_zoom_range, 1 + width_zoom_range]
            elif len(width_zoom_range) == 2:
                self.width_zoom_range = [width_zoom_range[0], width_zoom_range[1]]
            else:
                raise ValueError('`width_zoom_range` should be a float or a tuple or list of two floats. Received: %s' % (width_zoom_range,))
            if np.isscalar(height_zoom_range):
                self.height_zoom_range = [1 - height_zoom_range, 1 + height_zoom_range]
            elif len(height_zoom_range) == 2:
                self.height_zoom_range = [height_zoom_range[0], height_zoom_range[1]]
            else:
                raise ValueError('`width_zoom_range` should be a float or a tuple or list of two floats.  Received: %s' % (height_zoom_range,))
        else:
            self.width_zoom_range = None
            self.height_zoom_range = None
        if max_zoom_aspectratio<0:
            raise ValueError("max_zoom_aspectratio must be >=0")
        if min_zoom_aspectratio<0:
            raise ValueError("min_zoom_aspectratio must be >=0")
        if max_zoom_aspectratio<min_zoom_aspectratio:
            raise ValueError("min_zoom_aspectratio must be inferior to max_zoom_aspectratio")
        self.rotate90=rotate90
        self.max_zoom_aspectratio=max_zoom_aspectratio
        self.min_zoom_aspectratio=min_zoom_aspectratio
        self.min_histogram_range=min_histogram_range
        self.min_histogram_to_zero=min_histogram_to_zero
        self.noise_intensity=noise_intensity
        if np.isscalar(gaussian_blur_range):
            self.gaussian_blur_range=[gaussian_blur_range, gaussian_blur_range]
        else:
            self.gaussian_blur_range=gaussian_blur_range
        self.histogram_voodoo_n_points=histogram_voodoo_n_points
        self.histogram_voodoo_intensity=histogram_voodoo_intensity
        self.illumination_voodoo_n_points=illumination_voodoo_n_points
        self.illumination_voodoo_intensity=illumination_voodoo_intensity
        self.perform_illumination_augmentation = perform_illumination_augmentation
        self.bacteria_swim_distance=bacteria_swim_distance
        self.bacteria_swim_min_gap=bacteria_swim_min_gap
        self.closed_end = closed_end
        self.histogram_normalization_center=histogram_normalization_center
        self.histogram_normalization_scale=histogram_normalization_scale
        super().__init__(**kwargs)

    def get_random_transform(self, img_shape, seed=None):
        params = super().get_random_transform(img_shape, seed)
        if self.closed_end:
            if params.get("tx", 0)>0:
                params["tx"] = 0
            if params.get("flip_vertical", False):
                params["flip_vertical"]=False

        if self.width_zoom_range is not None:
            if self.width_zoom_range[0] == 1 and self.width_zoom_range[1] == 1:
                zy = 1
            else:
                zy = np.random.uniform(self.width_zoom_range[0], self.width_zoom_range[1])
            params['zy'] = zy
        if self.height_zoom_range is not None:
            if self.height_zoom_range[0] == 1 and self.height_zoom_range[1] == 1:
                zx = 1
            else:
                zx = np.random.uniform(self.height_zoom_range[0], self.height_zoom_range[1])
            params['zx'] = zx
        if self.max_zoom_aspectratio>0.: # ensure zx / zy < max_zoom_aspectratio
            if params['zx'] / params['zy'] > self.max_zoom_aspectratio:
                width_zoom_range = self.width_zoom_range if self.width_zoom_range else self.zoom_range
                height_zoom_range = self.height_zoom_range if self.height_zoom_range else self.zoom_range
                zx = params['zx']
                zy = params['zy']
                new_zx, new_zy = self._get_corrected_zoom_aspectratio(zx, zy, self.max_zoom_aspectratio, height_zoom_range, width_zoom_range)
                params['zx'] = new_zx
                params['zy'] = new_zy
                #print("max zoom aspect ration correction: old ar: {}/{}={}, new ar: {}/{}={}".format(zx, zy, zx/zy, params['zx'], params['zy'], params['zx'] / params['zy']))
        if self.min_zoom_aspectratio>0.: # ensure zx / zy > min_zoom_aspectratio
            if params['zx'] / params['zy'] < self.min_zoom_aspectratio:
                width_zoom_range = self.width_zoom_range if self.width_zoom_range else self.zoom_range
                height_zoom_range = self.height_zoom_range if self.height_zoom_range else self.zoom_range
                zx = params['zx']
                zy = params['zy']
                new_zx, new_zy = self._get_corrected_zoom_aspectratio(zx, zy, self.min_zoom_aspectratio, height_zoom_range, width_zoom_range)
                params['zx'] = new_zx
                params['zy'] = new_zy
                #print("min zoom aspect ration correction: old ar: {}/{}={}, new ar: {}/{}={}".format(zx, zy, zx/zy, params['zx'], params['zy'], params['zx'] / params['zy']))

        # limit rotation and horizontal translation according to max horizontal translation and zoom
        zx = params['zx']
        zy = params['zy']
        theta = params.get('theta')
        ty = params.get('ty')
        dy = tan(abs(theta)*pi/180) * img_shape[0] / (2 * zx) # max horizontal translation due to rotation
        # get max possible horizontal translation
        if np.isscalar(self.width_shift_range):
            if self.width_shift_range<1:
                max_ty = self.width_shift_range * img_shape[1]
            else:
                max_ty = self.width_shift_range
        else:
            max_ty = np.max(self.width_shift_range)

        delta = (dy + abs(ty)) - (max_ty * zy)
        if delta>0: # remove delta to both ty and theta
            if abs(ty) < delta/2:
                new_ty = 0
                delta -= abs(ty)
                new_theta = copysign(atan((dy - delta) * 2 * zx / img_shape[0]), theta) * 180 / pi
            elif dy < delta / 2:
                new_theta = 0
                delta -= dy
                new_ty = copysign(abs(ty) - delta, ty)
            else:
                new_ty = copysign(abs(ty) - delta/2, ty)
                new_theta = copysign(atan((dy - delta/2) * 2 * zx / img_shape[0]), theta) * 180 / pi
            params['ty'] = new_ty
            params['theta'] = new_theta
            #print("limit translation & rotation: ty: {}, dy: {} ty+dy {}, delta: {}, ty: {}->{}, theta: {}->{}, ".format(ty, dy, abs(ty)+dy, delta, ty, new_ty, theta, new_theta))
        if self.rotate90 and img_shape[0]==img_shape[1] and not getrandbits(1):
            params["rotate90"] = True
        # illumination parameters
        if self.perform_illumination_augmentation:
            if self.histogram_normalization_center is not None and self.histogram_normalization_scale is not None: # center / scale mode
                if isinstance(self.histogram_normalization_center, (list, tuple, np.ndarray)):
                    assert len(self.histogram_normalization_center)==2, "if histogram_normalization_center is a list/tuple it represent a range and should be of length 2"
                    params["center"] = uniform(self.histogram_normalization_center[0], self.histogram_normalization_center[1])
                else:
                    params["center"] = self.histogram_normalization_center
                if isinstance(self.histogram_normalization_scale, (list, tuple, np.ndarray)):
                    assert len(self.histogram_normalization_scale)==2, "if histogram_normalization_scale is a list/tuple it represent a range and should be of length 2"
                    params["scale"] = uniform(self.histogram_normalization_scale[0], self.histogram_normalization_scale[1])
                else:
                    params["scale"] = self.histogram_normalization_scale
            else: # min max mode
                if self.min_histogram_range<1 and self.min_histogram_range>0:
                    if self.min_histogram_to_zero:
                        params["vmin"] = 0
                        params["vmax"] = uniform(self.min_histogram_range, 1)
                    else:
                        vmin, vmax = pp.compute_histogram_range(self.min_histogram_range)
                        params["vmin"] = vmin
                        params["vmax"] = vmax
                elif self.min_histogram_range==1:
                    params["vmin"] = 0
                    params["vmax"] = 1
            if self.noise_intensity>0:
                poisson, speckle, gaussian = pp.get_random_noise_parameters(self.noise_intensity)
                params["poisson_noise"] = poisson
                params["speckle_noise"] = speckle
                params["gaussian_noise"] = gaussian
            if self.gaussian_blur_range[1]>0 and not getrandbits(1):
                params["gaussian_blur"] = uniform(self.gaussian_blur_range[0], self.gaussian_blur_range[1])

            if self.histogram_voodoo_n_points>0 and self.histogram_voodoo_intensity>0 and not getrandbits(1):
                # draw control points
                if "vmin" in params and "vmax" in params:
                    vmin = params["vmin"]
                    vmax = params["vmax"]
                    control_points = np.linspace(vmin, vmax, num=self.histogram_voodoo_n_points + 2)
                    target_points = pp.get_histogram_voodoo_target_points(control_points, self.histogram_voodoo_intensity)
                    params["histogram_voodoo_target_points"] = target_points
                elif "histogram_voodoo_target_points" in params:
                    del params["histogram_voodoo_target_points"]
            elif "histogram_voodoo_target_points" in params:
                del params["histogram_voodoo_target_points"]
            if self.illumination_voodoo_n_points>0 and self.illumination_voodoo_intensity>0 and not getrandbits(1):
                params["illumination_voodoo_target_points"] = pp.get_illumination_voodoo_target_points(self.illumination_voodoo_n_points, self.illumination_voodoo_intensity)
            elif "illumination_voodoo_target_points" in params:
                del params["illumination_voodoo_target_points"]
        return params

    def adjust_augmentation_param_from_mask(self, params, mask_img):
        if self.closed_end and params['zx']<1: # zoom in -> translate avoid upper part to be cut. this needs to be located here and not in get_random_transform because  cur zx is copied to prev / next
            params['tx'] = min(params.get("tx", 0), - mask_img.shape[0] * (1 - params['zx']) / 2 )
        forbid_transformations_if_object_touching_borders(params, mask_img, self.closed_end)
        if self.bacteria_swim_distance>1:
            # get y space between bacteria
            space_y = np.invert(np.any(mask_img, 1)).astype(np.int) # 1 where no label along y axis:
            space_y, n_lab = ndi.label(space_y)
            space_y = ndi.find_objects(space_y)
            space_y = [slice_obj[0] for slice_obj in space_y] # only first dim
            limit = mask_img.shape[0]
            space_y = [slice_obj for slice_obj in space_y if (slice_obj.stop - slice_obj.start)>=self.bacteria_swim_min_gap and slice_obj.stop>15 and (self.closed_end or slice_obj.start>0) and slice_obj.stop<limit] # keep only slices with length > 4 and not the space close to the open ends
            #space_y = [slice_obj for slice_obj in space_y if (slice_obj.stop - slice_obj.start)>2 and (self.closed_end or slice_obj.start>0) and slice_obj.stop<limit] # keep only slices with length > 2 and not the space close to the open ends
            if len(space_y)>0:
                space_y = [(slice_obj.stop + slice_obj.start)//2 for slice_obj in space_y]
                #space_y_mean = [(slice_obj.stop + slice_obj.start)//2 if slice_obj.start>0 else 0 for slice_obj in space_y]
                y = choice(space_y)
                #y = choice(space_y_mean)
                lower = self.closed_end or not getrandbits(1)
                if y==0:
                    lower = True
                # TODO choose a distance so that bacteria are not cut too much
                swim_params = {"y": y, "lower": lower, "distance": uniform(1, self.bacteria_swim_distance)}
                params["bacteria_swim"] = swim_params
                #params['tx'] = min_tx

    def adjust_augmentation_param_from_neighbor_mask(self, params, neighbor_mask_img):
        if params.get('zx', 1)>1: # forbid zx>1 if there are object @ border @ prev/next time point because zoom will be copied to prev/next transformation
            has_object_up, has_object_down = has_object_at_y_borders(neighbor_mask_img)
            if has_object_up or has_object_down:
                params["zx"] = 1

    def apply_transform(self, img, params):
        # swim before geom augmentation with because location must be precisely between 2 bacteria
        if "bacteria_swim" in params:
            y = params["bacteria_swim"]["y"]
            lower = params["bacteria_swim"]["lower"]
            dist = - params["bacteria_swim"]["distance"]
            if not lower:
                dist = - dist
            translation_params = {"tx" : dist}
            if lower:
                img[y:] = super().apply_transform(img[y:], translation_params)
            else:
                img[:y] = super().apply_transform(img[:y], translation_params)

        # geom augmentation
        img = super().apply_transform(img, params)
        if params.get("rotate90", False):
            img = np.rot90(img, k=1, axes=(0, 1))
        # illumination augmentation
        img = self._perform_illumination_augmentation(img, params)
        return img

    def _perform_illumination_augmentation(self, img, params):
        if "center" in params and "scale" in params:
            img = (img - params["center"]) / params["scale"]
        elif "vmin" in params and "vmax" in params:
            min = img.min()
            max = img.max()
            if min==max:
                raise ValueError("Image is blank, cannot perform illumination augmentation")
            img = pp.adjust_histogram_range(img, min=params["vmin"], max = params["vmax"], initial_range=[min, max])
        if "histogram_voodoo_target_points" in params:
            img = pp.histogram_voodoo(img, self.histogram_voodoo_n_points, self.histogram_voodoo_intensity, target_points = params["histogram_voodoo_target_points"])
        if "illumination_voodoo_target_points" in params:
            target_points = params["illumination_voodoo_target_points"]
            img = pp.illumination_voodoo(img, len(target_points), target_points=target_points)
        if params.get("gaussian_blur", 0)>0:
            img = pp.gaussian_blur(img, params["gaussian_blur"])
        if params.get("poisson_noise", 0)>0:
            img = pp.add_poisson_noise(img, params["poisson_noise"])
        if params.get("speckle_noise", 0)>0:
            img = pp.add_speckle_noise(img, params["speckle_noise"])
        if params.get("gaussian_noise", 0)>0:
            img = pp.add_gaussian_noise(img, params["gaussian_noise"])
        return img

    def _get_corrected_zoom_aspectratio(self, zx, zy, zoom_aspectratio, zx_range, zy_range):
        zx, zy = self._get_corrected_zoom_aspectratio1(zx, zy, zoom_aspectratio)
        if zx==1 and zy==1:
            return zx, zy
        if zx<zx_range[0]:
            zx = zx_range[0]
            zy = zx / zoom_aspectratio
            if zy<zy_range[0] or zy>zy_range[1]:
                zx, zy = 1, 1
        elif zx>zx_range[1]:
            zx = zx_range[1]
            zy = zx / zoom_aspectratio
            if zy<zy_range[0] or zy>zy_range[1]:
                zx, zy = 1, 1
        elif zy<zy_range[0]:
            zy = zy_range[0]
            zx = zy * zoom_aspectratio
            if zx<zx_range[0] or zx>zx_range[1]:
                zx, zy = 1, 1
        elif zy>zy_range[1]:
            zy = zy_range[1]
            zx = zy * zoom_aspectratio
            if zx<zx_range[0] or zx>zx_range[1]:
                zx, zy = 1, 1
        return zx, zy

    def _get_corrected_zoom_aspectratio1(self, zx, zy, zoom_aspectratio):
        if zx>=1 and zy>=1:
            if zoom_aspectratio==1:
                return 1, 1
            else:
                delta = (zoom_aspectratio * zy - zx) / (zoom_aspectratio - 1)
                return zx-delta, zy-delta
        elif zx>=1 and zy<1:
            delta = (zoom_aspectratio * zy - zx) / (- zoom_aspectratio - 1)
            return zx-delta, zy+delta
        elif zx<1 and zy>=1:
            delta = (zoom_aspectratio * zy - zx) / (zoom_aspectratio + 1)
            return zx+delta, zy-delta
        elif zx<1 and zy<1:
            if zoom_aspectratio==1:
                return 1, 1
            else:
                delta = (zoom_aspectratio * zy - zx) / (- zoom_aspectratio + 1)
                return zx+delta, zy+delta
        return delta

def has_object_at_y_borders(mask_img):
    return np.any(mask_img[[-1,0], :], 1) # np.flip()

def forbid_transformations_if_object_touching_borders(aug_param, mask_img, closed_end):
    tx = aug_param.get('tx', 0)
    zx = aug_param.get('zx', 1)
    if tx!=0 or zx>1:
        has_object_up, has_object_down = has_object_at_y_borders(mask_img) # up & down as in the displayed image
        if zx<1: # zoom in
            tx_lim = mask_img.shape[0] * (1 - zx) / 2
        else:
            tx_lim = 0
        if (has_object_down and has_object_up):
            if closed_end:
                aug_param['tx']=-tx_lim
            else:
                aug_param['tx']=0
        elif has_object_down and tx<-tx_lim:
            aug_param['tx']=-tx_lim
        elif has_object_up and tx>tx_lim:
            aug_param['tx']=tx_lim
        if has_object_up or has_object_down:
            if zx>1:
                aug_param['zx'] = 1
