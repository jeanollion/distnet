from keras_preprocessing.image import ImageDataGenerator
import numpy as np
from math import tan, atan, pi, copysign
import dlutils.pre_processing_utils as pp
from random import getrandbits, uniform
import copy

class ImageDataGeneratorMM(ImageDataGenerator):
    def __init__(self, width_zoom_range=0., height_zoom_range=0., max_zoom_aspectratio=1.5, min_zoom_aspectratio=0., perform_illumination_augmentation = True, gaussian_blur_range=[1, 2], noise_intensity = 0.1, min_histogram_range=0.1, histogram_voodoo_n_points=5, histogram_voodoo_intensity=0.5, illumination_voodoo_n_points=5, illumination_voodoo_intensity=0.6, **kwargs):
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
        self.max_zoom_aspectratio=max_zoom_aspectratio
        self.min_zoom_aspectratio=min_zoom_aspectratio
        self.min_histogram_range=min_histogram_range
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
        super().__init__(**kwargs)

    def get_random_transform(self, img_shape, seed=None):
        params = super().get_random_transform(img_shape, seed)
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

        # illumination parameters
        if self.perform_illumination_augmentation:
            if self.min_histogram_range<1:
                vmin, vmax = pp.compute_histogram_range(self.min_histogram_range)
                params["vmin"] = vmin
                params["vmax"] = vmax
            else:
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
                min = params["vmin"]
                max = params["vmax"]
                control_points = np.linspace(min, max, num=self.histogram_voodoo_n_points + 2)
                target_points = pp.get_histogram_voodoo_target_points(control_points, self.histogram_voodoo_intensity)
                params["histogram_voodoo_target_points"] = target_points
            if self.illumination_voodoo_n_points>0 and self.illumination_voodoo_intensity>0 and not getrandbits(1):
                params["illumination_voodoo_target_points"] = pp.get_illumination_voodoo_target_points(self.illumination_voodoo_n_points, self.illumination_voodoo_intensity)
        return params

    def apply_transform(self, img, params):
        img = super().apply_transform(img, params)
        #print("parameters:", params)
        if "vmin" in params and "vmax" in params:
            min = img.min()
            max = img.max()
            if min==max:
                raise ValueError("Image is blank, cannot perform illumination augmentation")
            img = pp.adjust_histogram_range(img, min=params["vmin"], max = params["vmax"], initial_range=[min, max])
        if "histogram_voodoo_target_points" in params:
            target_points = params["histogram_voodoo_target_points"]
            img = pp.histogram_voodoo(img, len(target_points) - 2, target_points = target_points)
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
            img = pp.add_speckle_noise(img, params["gaussian_noise"])
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

def transfer_illumination_aug_parameters(source, dest):
    if "vmin" in source and "vmax" in source:
        dest["vmin"] = source["vmin"]
        dest["vmax"] = source["vmax"]
    else:
        if "vmin" in dest:
            del dest["vmin"]
        if "vmax" in dest:
            del des["vmax"]
    if "poisson_noise" in source:
        dest["poisson_noise"] = source.get("poisson_noise", 0)
    elif "poisson_noise" in dest:
        del dest["poisson_noise"]
    if "speckle_noise" in source:
        dest["speckle_noise"] = source.get("speckle_noise", 0)
    elif "speckle_noise" in dest:
        del dest["speckle_noise"]
    if "gaussian_noise" in source:
        dest["gaussian_noise"] = source.get("gaussian_noise", 0)
    elif "gaussian_noise" in dest:
        del dest["gaussian_noise"]
    if "gaussian_blur" in source:
        dest["gaussian_blur"] = source.get("gaussian_blur", 0)
    elif "gaussian_blur" in dest:
        del dest["gaussian_blur"]
    if "histogram_voodoo_target_points" in source:
        dest["histogram_voodoo_target_points"] = copy.copy(source["histogram_voodoo_target_points"])
    elif "histogram_voodoo_target_points" in dest:
        del dest["histogram_voodoo_target_points"]
    if "illumination_voodoo_target_points" in source:
        dest["illumination_voodoo_target_points"] = copy.copy(source["illumination_voodoo_target_points"])
    elif "illumination_voodoo_target_points" in dest:
        del dest["illumination_voodoo_target_points"]
