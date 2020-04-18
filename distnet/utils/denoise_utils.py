# adapted from noise2self https://github.com/czbiohub/noise2self
import numpy as np
from scipy.ndimage import convolve
from numpy.random import randint
import itertools

METHOD = ["ZERO", "AVERAGE", "RANDOM"]

def get_denoiser_manipulation_fun(method=METHOD[2], patch_size=3, random_patch_radius=1):
    if method==METHOD[0]:
        def fun(batch):
            image_shape = batch.shape[1:-1]
            offset = get_random_offset(patch_size)
            mask_coords = get_mask_coords(patch_size , offset, image_shape)
            output = get_output(batch, mask_coords)
            for b,c in itertools.product(range(batch.shape[0]), range(batch.shape[-1])):
                mask_idx = (b,) + mask_coords + (c,)
                batch[mask_idx] = 0
            return output
        return fun
    elif method==METHOD[1]:
        def fun(batch):
            image_shape = batch.shape[1:-1]
            offset = get_random_offset(patch_size)
            mask_coords = get_mask_coords(patch_size , offset, image_shape)
            output = get_output(batch, mask_coords)
            avg = average_batch(batch)
            for b,c in itertools.product(range(batch.shape[0]), range(batch.shape[-1])):
                mask_idx = (b,) + mask_coords + (c,)
                batch[mask_idx] = avg[mask_idx]
            return output
        return fun
    elif method==METHOD[2]:
        def fun(batch):
            offset = get_random_offset(patch_size)
            image_shape = batch.shape[1:-1]
            mask_coords = get_mask_coords(patch_size , offset, image_shape)
            output = get_output(batch, mask_coords)
            if len(image_shape)==2:
                get_random_coords = get_random_coords2D
            else:
                raise ValueError("Image Rank not supported yet")
            for b,c in itertools.product(range(batch.shape[0]), range(batch.shape[-1])):
                replacement_coords = get_random_coords(random_patch_radius, mask_coords, image_shape)
                mask_idx = (b,) + mask_coords + (c,)
                replacement_idx = (b,) + replacement_coords + (c,)
                batch[mask_idx] = batch[replacement_idx]
            return output
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

def get_random_offset(patch_size):
    grid_offset = randint(0, patch_size**2)
    offset_x = grid_offset % patch_size
    offset_y = (grid_offset // patch_size) % patch_size
    return offset_y, offset_x

def pixel_grid_mask(shape, patch_size, offset_y, offset_x):
    mask = np.zeros(shape)
    for y in range(shape[0]):
        for x in range(shape[1]):
            if (x % patch_size == offset_x and y % patch_size == offset_y):
                mask[y, x] = 1
    return mask

def get_mask_coords(patch_size, offset, shape):
    if len(offset)!=len(shape):
        raise ValueError("offset and shape must have same rank")
    coords = [np.arange(int(np.ceil((shape[i]-offset[i]) / patch_size))) * patch_size + offset[i] for i in range(len(shape))]
    return tuple([a.flatten() for a in np.meshgrid(*coords, sparse=False, indexing='ij')])

def get_random_coords2D(patch_radius, coords_yx, shape):
    patch_size = 2 * patch_radius + 1
    n_coords = patch_size**2
    center = n_coords // 2
    choices = list(range(0,center)) + list(range(center+1, n_coords))
    yx = np.random.choice(choices, size=coords_yx[0].shape[0], replace=True)
    y = yx % patch_size - patch_radius
    x = ((yx // patch_size) % patch_size) - patch_radius
    y += coords_yx[0]
    x += coords_yx[1]
    # mirror coords outside image (center on coord to avoid targeting center)
    mask_y = (y<0) | (y>=shape[0])
    y[mask_y] = 2 * coords_yx[0][mask_y] - y[mask_y]
    mask_x = (x<0) | (x>=shape[1])
    x[mask_x] = 2 * coords_yx[1][mask_x] - x[mask_x]
    return (y, x)

AVG_KERNEL = np.array([[0.5/6, 1.0/6, 0.5/6], [1.0/6, 0.0, 1.0/6], (0.5/6, 1.0/6, 0.5/6)])

def average_batch(batch):
    ker = AVG_KERNEL[np.newaxis, :, :, np.newaxis]
    return convolve(batch, ker, mode ="mirror")
