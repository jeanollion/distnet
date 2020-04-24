# adapted from noise2self https://github.com/czbiohub/noise2self
import numpy as np
from scipy.ndimage import convolve
from numpy.random import randint
import itertools
from .helpers import ensure_multiplicity

METHOD = ["ZERO", "AVERAGE", "RANDOM"]

def get_denoiser_manipulation_fun(method=METHOD[2], patch_shape=3, random_patch_radius=1, mask_X_radius=0):
    if method==METHOD[0]:
        def fun(batch):
            image_shape = batch.shape[1:-1]
            patch_shape_ = ensure_multiplicity(len(image_shape), patch_shape)
            if isinstance(patch_shape_, list):
                patch_shape_ = tuple(patch_shape_)
            offset = get_random_offset(patch_shape_)
            mask_coords = get_mask_coords(patch_shape_, offset, image_shape)
            output = get_output(batch, mask_coords)
            if mask_X_radius>0:
                mask_coords = get_extended_mask_coordsX(mask_coords, min(patch_shape[-1]//2, mask_X_radius), image_shape)
            for b,c in itertools.product(range(batch.shape[0]), range(batch.shape[-1])):
                mask_idx = (b,) + mask_coords + (c,)
                batch[mask_idx] = 0
            return output
        return fun
    elif method==METHOD[1]:
        def fun(batch):
            image_shape = batch.shape[1:-1]
            patch_shape_ = ensure_multiplicity(len(image_shape), patch_shape)
            if isinstance(patch_shape_, list):
                patch_shape_ = tuple(patch_shape_)
            offset = get_random_offset(patch_shape_)
            mask_coords = get_mask_coords(patch_shape_ , offset, image_shape)
            output = get_output(batch, mask_coords)
            avg = average_batch(batch, exclude_X=mask_X_radius>0)
            if mask_X_radius>0:
                mask_coords = get_extended_mask_coordsX(mask_coords, min(patch_shape[-1]//2, mask_X_radius), image_shape)
            for b,c in itertools.product(range(batch.shape[0]), range(batch.shape[-1])):
                mask_idx = (b,) + mask_coords + (c,)
                batch[mask_idx] = avg[mask_idx]
            return output
        return fun
    elif method==METHOD[2]:
        def fun(batch):
            image_shape = batch.shape[1:-1]
            patch_shape_ = ensure_multiplicity(len(image_shape), patch_shape)
            if isinstance(patch_shape_, list):
                patch_shape_ = tuple(patch_shape_)
            offset = get_random_offset(patch_shape_)
            r_patch_radius = ensure_multiplicity(len(image_shape), random_patch_radius)
            if isinstance(r_patch_radius, list):
                r_patch_radius = tuple(r_patch_radius)
            mask_coords = get_mask_coords(patch_shape_ , offset, image_shape)
            output = get_output(batch, mask_coords)
            if mask_X_radius>0:
                mask_coords = get_extended_mask_coordsX(mask_coords, min(patch_shape[-1]//2, mask_X_radius), image_shape)
            for b,c in itertools.product(range(batch.shape[0]), range(batch.shape[-1])):
                replacement_coords = get_random_coords(r_patch_radius, mask_coords, image_shape, exclude_X = mask_X_radius>0)
                mask_idx = (b,) + mask_coords + (c,)
                replacement_idx = (b,) + replacement_coords + (c,)
                batch[mask_idx] = batch[replacement_idx]
            return output
        return fun
    elif method=="TEST":
        def fun(batch):
            image_shape = batch.shape[1:-1]
            patch_shape_ = ensure_multiplicity(len(image_shape), patch_shape)
            if isinstance(patch_shape_, list):
                patch_shape_ = tuple(patch_shape_)
            offset = get_random_offset(patch_shape_)
            r_patch_radius = ensure_multiplicity(len(image_shape), random_patch_radius)
            if isinstance(r_patch_radius, list):
                r_patch_radius = tuple(r_patch_radius)
            mask_coords = get_mask_coords(patch_shape_ , offset, image_shape)
            if mask_X_radius>0:
                mask_coords = get_extended_mask_coordsX(mask_coords, min(patch_shape[-1]//2, mask_X_radius), image_shape)
            mask = np.zeros(batch.shape, dtype=batch.dtype)
            mask2 = np.zeros(batch.shape, dtype=batch.dtype)
            mask_values = np.arange(mask_coords[0].shape[0])
            np.random.shuffle(mask_values)
            for b,c in itertools.product(range(batch.shape[0]), range(batch.shape[-1])):
                replacement_coords = get_random_coords(r_patch_radius, mask_coords, image_shape, exclude_X=mask_X_radius>0)
                mask_idx = (b,) + mask_coords + (c,)
                mask[mask_idx] = mask_values
                replacement_idx = (b,) + replacement_coords + (c,)
                mask2[replacement_idx] = mask_values
            return np.concatenate([mask2, mask], axis=-1)
        return fun
    else:
        raise ValueError("Invalid method")

def get_output(batch, mask_coords):
    mask = np.zeros(batch.shape, dtype=batch.dtype)
    n_pix = float(np.prod(batch.shape[1:-1]))
    mask_value = n_pix / len(mask_coords[0])
    for b,c in itertools.product(range(batch.shape[0]), range(batch.shape[-1])):
        mask_idx = (b,) + mask_coords + (c,)
        mask[mask_idx] = mask_value
    return np.concatenate([batch, mask], axis=-1)

def get_random_offset(patch_shape):
    grid_offset = randint(0, np.product(np.array(patch_shape)))
    return np.unravel_index(grid_offset, patch_shape)

def get_mask_coords(patch_shape, offset, img_shape):
    if len(offset)!=len(img_shape):
        raise ValueError("offset and shape must have same rank")
    coords = [np.arange(int(np.ceil((img_shape[i]-offset[i]) / patch_shape[i]))) * patch_shape[i] + offset[i] for i in range(len(img_shape))]
    return tuple([a.flatten() for a in np.meshgrid(*coords, sparse=False, indexing='ij')])

def get_extended_mask_coordsX(mask_coords, radX, img_shape):
    if radX==0:
        return mask_coords
    extended_coords = []
    rank = len(img_shape)
    for dX in range(1, radX+1):
        # right
        coords = [np.copy(mask_coords[ax]) for ax in range(rank)]
        coords[-1] += dX
        mask = coords[-1]<img_shape[-1] # remove out-of-bound
        coords = [coords[ax][mask] for ax in range(rank)]
        extended_coords.append(coords)
        # left
        coords = [np.copy(mask_coords[ax]) for ax in range(rank)]
        coords[-1] -= dX
        mask = coords[-1]>=0 # remove out-of-bound
        coords = [coords[ax][mask] for ax in range(rank)]
        extended_coords.append(coords)

    result = []
    for ax in range(rank):
        to_concat = [extended_coords[i][ax] for i in range(len(extended_coords))]
        result.append( np.concatenate([mask_coords[ax]]+to_concat))
    return tuple(result)

def get_random_coords(patch_radius, offsets, img_shape, exclude_X = False):
    patch_shape = 2 * np.array(patch_radius) + 1
    n_coords = np.product(patch_shape)
    patch_shape = tuple(patch_shape)
    center = n_coords // 2
    if not exclude_X:
        choices = list(range(0,center)) + list(range(center+1, n_coords))
    else:
        choices = [i for i in range(n_coords) if np.any(np.unravel_index(i, patch_shape)[:-1]!=patch_radius[:-1])]
    indices = np.random.choice(choices, size=offsets[0].shape[0], replace=True)
    coords = list(np.unravel_index(indices, patch_shape))
    for axis in range(len(offsets)):
        coords[axis] += offsets[axis] - patch_radius[axis]
        # mirror coords outside image (center on coord to avoid targeting center)
        mask = (coords[axis]<0) | (coords[axis]>=img_shape[axis])
        coords[axis][mask] = 2 * offsets[axis][mask] - coords[axis][mask]
        #coords[axis][mask] = 0
    return tuple(coords)

AVG_KERNEL_1D = np.array([1.0/2, 0.0, 1.0/2])[np.newaxis, :, np.newaxis]
AVG_KERNEL_2D = np.array([[0.5/6, 1.0/6, 0.5/6], [1.0/6, 0.0, 1.0/6], [0.5/6, 1.0/6, 0.5/6]])[np.newaxis, :, :, np.newaxis]
AVG_KERNEL_3D = np.array([
    [[0.5/16, 0.5/16, 0.5/16], [0.5/16, 1.0, 0.5/16], [0.5/16, 0.5/16, 0.5/16]],
    [[0.5/16, 1.0/16, 0.5/16], [1.0/16, 0.0, 1.0/16], [0.5/16, 1.0/16, 0.5/16]],
    [[0.5/16, 0.5/16, 0.5/16], [0.5/16, 1.0, 0.5/16], [0.5/16, 0.5/16, 0.5/16]]
])[np.newaxis, :, :, :, np.newaxis]
AVG_KERNEL_2D_X = np.array([[0.5/4, 1.0/4, 0.5/4], [0.0, 0.0, 0.0], (0.5/4, 1.0/4, 0.5/4)])[np.newaxis, :, :, np.newaxis]
AVG_KERNEL_3D_X = np.array([
    [[0.5/14, 0.5/14, 0.5/14], [0.5/14, 1.0, 0.5/14], [0.5/14, 0.5/14, 0.5/14]],
    [[0.5/14, 1.0/14, 0.5/14], [0.0, 0.0, 0.0], [0.5/14, 1.0/14, 0.5/14]],
    [[0.5/14, 0.5/14, 0.5/14], [0.5/14, 1.0, 0.5/14], [0.5/14, 0.5/14, 0.5/14]]
])[np.newaxis, :, :, :, np.newaxis]

def average_batch(batch, exclude_X=False):
    rank = batch.ndim - 2 # exclude batch & channel
    if rank==2:
        ker = AVG_KERNEL_2D if not exclude_X else AVG_KERNEL_2D_X
    elif rank==3:
        ker = AVG_KERNEL_3D if not exclude_X else AVG_KERNEL_3D_X
    elif rank==1:
        if exclude_X:
            raise ValueError("exclude_X incopatible with 1D arrays")
        ker = AVG_KERNEL_1D
    else:
        raise ValueError("Only 1D, 2D or 3D arrays supported")
    return convolve(batch, ker, mode ="mirror")
