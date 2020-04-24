import numpy as np
from scipy.ndimage import convolve
from scipy.stats import multivariate_normal
from numpy.random import randint
import itertools
from .helpers import ensure_multiplicity

METHOD = ["ZERO", "AVERAGE", "RANDOM"]

def get_denoiser_manipulation_fun(method=METHOD[1], grid_shape=3, grid_random_increase_shape=0, average_radius = 1, random_patch_radius=1, mask_X_radius=0):
    if method==METHOD[0]:
        def fun(batch):
            image_shape = batch.shape[1:-1]
            grid_shape_ = ensure_multiplicity(len(image_shape), grid_shape)
            grid_shape_ = random_increase_grid_shape(grid_random_increase_shape, grid_shape_)
            offset = get_random_offset(grid_shape_)
            mask_coords = get_mask_coords(grid_shape_, offset, image_shape)
            output = get_output(batch, mask_coords)
            if mask_X_radius>0:
                mask_coords = get_extended_mask_coordsX(mask_coords, min(grid_shape[-1]//2, mask_X_radius), image_shape)
            for b,c in itertools.product(range(batch.shape[0]), range(batch.shape[-1])):
                mask_idx = (b,) + mask_coords + (c,)
                batch[mask_idx] = 0
            return output
        return fun
    elif method==METHOD[1]:
        if average_radius not in [1, 2, 3]:
            raise ValueError("Average radius must be in [1, 2, 3]")
        def fun(batch):
            image_shape = batch.shape[1:-1]
            grid_shape_ = ensure_multiplicity(len(image_shape), grid_shape)
            grid_shape_ = random_increase_grid_shape(grid_random_increase_shape, grid_shape_)
            offset = get_random_offset(grid_shape_)
            mask_coords = get_mask_coords(grid_shape_ , offset, image_shape)
            output = get_output(batch, mask_coords)
            avg = average_batch(batch, radius = average_radius, exclude_X=mask_X_radius>0)
            if mask_X_radius>0:
                mask_coords = get_extended_mask_coordsX(mask_coords, min(grid_shape[-1]//2, mask_X_radius), image_shape)
            for b,c in itertools.product(range(batch.shape[0]), range(batch.shape[-1])):
                mask_idx = (b,) + mask_coords + (c,)
                batch[mask_idx] = avg[mask_idx]
            return output
        return fun
    elif method==METHOD[2]:
        def fun(batch):
            image_shape = batch.shape[1:-1]
            grid_shape_ = ensure_multiplicity(len(image_shape), grid_shape)
            grid_shape_ = random_increase_grid_shape(grid_random_increase_shape, grid_shape_)
            offset = get_random_offset(grid_shape_)
            r_patch_radius = ensure_multiplicity(len(image_shape), random_patch_radius)
            if isinstance(r_patch_radius, list):
                r_patch_radius = tuple(r_patch_radius)
            mask_coords = get_mask_coords(grid_shape_ , offset, image_shape)
            output = get_output(batch, mask_coords)
            if mask_X_radius>0:
                mask_coords = get_extended_mask_coordsX(mask_coords, min(grid_shape[-1]//2, mask_X_radius), image_shape)
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
            grid_shape_ = ensure_multiplicity(len(image_shape), grid_shape)
            grid_shape_ = random_increase_grid_shape(grid_random_increase_shape, grid_shape_)
            offset = get_random_offset(grid_shape_)
            r_patch_radius = ensure_multiplicity(len(image_shape), random_patch_radius)
            if isinstance(r_patch_radius, list):
                r_patch_radius = tuple(r_patch_radius)
            mask_coords = get_mask_coords(grid_shape_ , offset, image_shape)
            if mask_X_radius>0:
                mask_coords = get_extended_mask_coordsX(mask_coords, min(grid_shape[-1]//2, mask_X_radius), image_shape)
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

def random_increase_grid_shape(random_increase_shape, grid_shape):
    shape_increase = ensure_multiplicity(len(grid_shape), random_increase_shape)
    return tuple([grid_shape[ax] + randint(0, high=shape_increase[ax]+1) if shape_increase[ax]>0 else grid_shape[ax] for ax in range(len(grid_shape))])

def get_random_offset(grid_shape):
    grid_offset = randint(0, np.product(np.array(grid_shape)))
    return np.unravel_index(grid_offset, grid_shape)

def get_mask_coords(grid_shape, offset, img_shape):
    if len(offset)!=len(img_shape):
        raise ValueError("offset and shape must have same rank")
    coords = [np.arange(int(np.ceil((img_shape[i]-offset[i]) / grid_shape[i]))) * grid_shape[i] + offset[i] for i in range(len(img_shape))]
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
    grid_shape = 2 * np.array(patch_radius) + 1
    n_coords = np.product(grid_shape)
    grid_shape = tuple(grid_shape)
    center = n_coords // 2
    if not exclude_X:
        choices = list(range(0,center)) + list(range(center+1, n_coords))
    else:
        choices = [i for i in range(n_coords) if np.any(np.unravel_index(i, grid_shape)[:-1]!=patch_radius[:-1])]
    indices = np.random.choice(choices, size=offsets[0].shape[0], replace=True)
    coords = list(np.unravel_index(indices, grid_shape))
    for axis in range(len(offsets)):
        coords[axis] += offsets[axis] - patch_radius[axis]
        # mirror coords outside image (center on coord to avoid targeting center)
        mask = (coords[axis]<0) | (coords[axis]>=img_shape[axis])
        coords[axis][mask] = 2 * offsets[axis][mask] - coords[axis][mask]
        #coords[axis][mask] = 0
    return tuple(coords)

def get_nd_gaussian_kernel(radius=1, sigma=0, ndim=2):
    size = 2 * radius + 1
    if ndim == 1:
        coords = [np.mgrid[-radius:radius:complex(0, size)]]
    elif ndim==2:
        coords = np.mgrid[-radius:radius:complex(0, size), -radius:radius:complex(0, size)]
    elif ndim==3:
        coords = np.mgrid[-radius:radius:complex(0, size), -radius:radius:complex(0, size), -radius:radius:complex(0, size)]
    else:
        raise ValueError("Up to 3D supported")

    # Need an (N, ndim) array of coords pairs.
    stacked = np.column_stack([c.flat for c in coords])
    mu = np.array([0.0]*ndim)
    s = np.array([sigma if sigma>0 else radius]*ndim)
    covariance = np.diag(s**2)
    z = multivariate_normal.pdf(stacked, mean=mu, cov=covariance)
    # Reshape back to a (30, 30) grid.
    z = z.reshape(coords[0].shape)
    return z/z.sum()

def get_nd_gaussian_donut_kernel(radius=1, sigma=0, ndim=2, exclude_X=False):
    ker = get_nd_gaussian_kernel(radius, sigma, ndim)
    if exclude_X:
        if ndim==1:
            raise ValueError("exclude_X incopatible with 1D arrays")
        elif ndim==2:
            for x in range(-radius, radius+1):
                ker[radius, x] = 0
        elif ndim==3:
            for x in range(-radius, radius+1):
                ker[radius, radius, x] = 0
    else:
        if ndim==1:
            ker[radius] = 0
        elif ndim==2:
            ker[radius, radius] = 0
        elif ndim==3:
            ker[radius, radius, radius] = 0
    return ker / ker.sum()

AVG_KERNELS_1D = {r:get_nd_gaussian_donut_kernel(r, ndim=1)[np.newaxis, ..., np.newaxis] for r in [1, 2, 3]}
AVG_KERNELS_2D = {r:get_nd_gaussian_donut_kernel(r, ndim=2)[np.newaxis, ..., np.newaxis] for r in [1, 2, 3]}
AVG_KERNELS_3D = {r:get_nd_gaussian_donut_kernel(r, ndim=3)[np.newaxis, ..., np.newaxis] for r in [1, 2, 3]}
AVG_KERNELS_2D_X = {r:get_nd_gaussian_donut_kernel(r, ndim=2, exclude_X=True)[np.newaxis, ..., np.newaxis] for r in [1, 2, 3]}
AVG_KERNELS_3D_X = {r:get_nd_gaussian_donut_kernel(r, ndim=3, exclude_X=True)[np.newaxis, ..., np.newaxis] for r in [1, 2, 3]}

def average_batch(batch, radius=1, exclude_X=False):
    rank = batch.ndim - 2 # exclude batch & channel
    if rank==2:
        ker = AVG_KERNELS_2D[radius] if not exclude_X else AVG_KERNELS_2D_X[radius]
    elif rank==3:
        ker = AVG_KERNELS_3D[radius] if not exclude_X else AVG_KERNELS_3D_X[radius]
    elif rank==1:
        if exclude_X:
            raise ValueError("exclude_X incopatible with 1D arrays")
        ker = AVG_KERNELS_1D[radius]
    else:
        raise ValueError("Only 1D, 2D or 3D arrays supported")
    return convolve(batch, ker, mode ="mirror")
