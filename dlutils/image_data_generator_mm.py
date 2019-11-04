from keras_preprocessing.image import ImageDataGenerator
import numpy as np
from math import tan, atan, pi, copysign

class ImageDataGeneratorMM(ImageDataGenerator):
    def __init__(self, width_zoom_range=0., height_zoom_range=0., max_zoom_aspectratio=1.5, min_zoom_aspectratio=0., **kwargs):
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
        return params

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
