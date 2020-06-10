import numpy as np
from scipy.ndimage import convolve, median_filter, generic_filter
from scipy.stats import multivariate_normal
from numpy.random import randint
import itertools
from .helpers import ensure_multiplicity
#from robustats import weighted_median

METHOD = ["AVERAGE", "RANDOM"]

def get_blind_spot_masking_fun(method=METHOD[0], grid_shape=3, grid_random_increase_shape=2, radius = 1, mask_X_radius=0, drop_grid_proportion = 0.1, grid_random_channel = True, **kwargs):
    """masking function for self-supervised denoising.

    Parameters
    ----------
    method : string within ["AVERAGE", "RANDOM"]
        masking method:
            "AVERAGE" replace pixels by a gaussian average of the neighboring pixels (excluding the center pixel). radius controls the radius of the gaussian
            "RANDOM" replace pixels by a randomly selected pixel around the pixel (excluding the center pixel). radius controls the size of the patch where random pixels are selected

    grid_shape : integer / tuple of integer
        controls grid space
    grid_random_increase_shape : integer / tuple of integer
        if greater than 0 a random value within the range [0, grid_random_increase_shape] will be added to the grid spacing so that grid spacing is in [grid_shape, grid_shape+grid_random_increase_shape]
    radius : integer / tuple of integer
        radius of the gaussian filter for AVERAGE method or of the patch to select random pixels for RANDOM method
    mask_X_radius : integer
        for structured noise: also mask horizontal pixels within the range [-mask_X_radius, mask_X_radius]

    Returns
    -------
    function
        function that inputs the batch (format NYXC), mask some of the batch pixels along a grid, and returns an output batch with shape N,Y,X,2*C
        the first C channels correspond to the input batch and the last C channels correspond to the masking grid, where the value is 0 outside the grid, and X*Y / n_pix where n_pix is the number of masked pixels
    """
    if method==METHOD[0] or method=="MEDIAN" or method=="WMEDIAN":
        def fun(batch):
            image_shape = batch.shape[1:-1]
            n_pix = float(np.prod(image_shape))
            grid_shape_ = ensure_multiplicity(len(image_shape), grid_shape)

            grid_shape_r = random_increase_grid_shape(grid_random_increase_shape, grid_shape_)
            offset = get_random_offset(grid_shape_r)
            mask_coords = get_mask_coords(grid_shape_r, offset, image_shape)
            if drop_grid_proportion>0:
                mask_coords = remove_random_grid_points(mask_coords, drop_grid_proportion)
            output = get_output(batch, mask_coords, random_channel=grid_random_channel)
            if mask_X_radius>0:
                mask_coords = get_extended_mask_coordsX(mask_coords, mask_X_radius, image_shape)
            if method==METHOD[0]:
                avg = average_batch(batch, radius = radius, exclude_X=mask_X_radius>0)
                r_values = lambda b,c : avg[b,...,c][mask_coords]
            else: # median. only computed on grid
                grid_exp = expand_coordinates(mask_coords, radius, image_shape)
                if method == "MEDIAN":
                    r_values = lambda b,c : median_coord(batch[b,...,c], grid_exp, exclude_X=mask_X_radius>0)
                else:
                    r_values = lambda b,c : weighted_median_coord(batch[b,...,c], grid_exp, exclude_X=mask_X_radius>0)
            for b, c in itertools.product(range(batch.shape[0]), range(batch.shape[-1])):
                batch[b,...,c][mask_coords] = r_values(b,c) # masking
            return output
        return fun
    elif method==METHOD[1]:
        def fun(batch):
            image_shape = batch.shape[1:-1]
            n_pix = float(np.prod(image_shape))
            grid_shape_ = ensure_multiplicity(len(image_shape), grid_shape)
            r_patch_radius = ensure_multiplicity(len(image_shape), radius)
            if isinstance(r_patch_radius, list):
                r_patch_radius = tuple(r_patch_radius)
            grid_shape_r = random_increase_grid_shape(grid_random_increase_shape, grid_shape_)
            offset = get_random_offset(grid_shape_r)
            mask_coords = get_mask_coords(grid_shape_r , offset, image_shape)
            if drop_grid_proportion>0:
                mask_coords = remove_random_grid_points(mask_coords, drop_grid_proportion)
            output = get_output(batch, mask_coords, random_channel=grid_random_channel)
            if mask_X_radius>0:
                mask_coords = get_extended_mask_coordsX(mask_coords, mask_X_radius, image_shape)
            replacement_coords = get_random_coords(r_patch_radius, mask_coords, image_shape, exclude_X = mask_X_radius>0)
            for b, c in itertools.product(range(batch.shape[0]), range(batch.shape[-1])):
                m_coords = (b,) + mask_coords + (c,)
                r_coords = (b,) + replacement_coords + (c,)
                batch[m_coords] = batch[r_coords] # masking
            return output
        return fun
    elif method=="FUNCTION": # THIS RETURNS A channels_postprocessing_function -> input is a dict with all channels. additional arguments : input_channel_idx / output_channel_idx / mean_channel_idx (facultative) / sigma_channel_idx
        def fun(batch_per_channel):
            assert isinstance(batch_per_channel, dict), "input must be a dict mapping channel indices to corresponding batches. in FUNCTION method the returned function should be used as a channels_postprocessing_function"
            i_idx = kwargs["input_channel_idx"]
            batch = batch_per_channel[i_idx]
            o_idx = kwargs["output_channel_idx"]
            replacement_function = kwargs["replacement_function"]

            image_shape = batch.shape[1:-1]
            n_pix = float(np.prod(image_shape))
            grid_shape_ = ensure_multiplicity(len(image_shape), grid_shape)

            grid_shape_r = random_increase_grid_shape(grid_random_increase_shape, grid_shape_)
            offset = get_random_offset(grid_shape_r)
            mask_coords = get_mask_coords(grid_shape_r, offset, image_shape)
            if drop_grid_proportion>0:
                mask_coords = remove_random_grid_points(mask_coords, drop_grid_proportion)
            batch_per_channel[o_idx] = get_output(batch, mask_coords, random_channel=grid_random_channel)
            if mask_X_radius>0:
                mask_coords = get_extended_mask_coordsX(mask_coords, mask_X_radius, image_shape)
            r_values = replacement_function(batch_per_channel, mask_coords)
            for b, c in itertools.product(range(batch.shape[0]), range(batch.shape[-1])):
                batch[b,...,c][mask_coords] = r_values[b,c] # masking
        return fun
    elif method=="CONSTANT" or method=="MAXIMUM":
        def fun(batch):
            image_shape = batch.shape[1:-1]
            n_pix = float(np.prod(image_shape))
            grid_shape_ = ensure_multiplicity(len(image_shape), grid_shape)
            r_patch_radius = ensure_multiplicity(len(image_shape), radius)
            if isinstance(r_patch_radius, list):
                r_patch_radius = tuple(r_patch_radius)
            constant_replacement_value = kwargs.get(constant_replacement_value, None)
            if constant_replacement_value is None or method=="MAXIMUM":
                constant_replacement_value_ = np.max(batch)
            elif callable(constant_replacement_value):
                constant_replacement_value_ = constant_replacement_value(batch)
            else:
                constant_replacement_value_ = constant_replacement_value
            grid_shape_r = random_increase_grid_shape(grid_random_increase_shape, grid_shape_)
            offset = get_random_offset(grid_shape_r)
            mask_coords = get_mask_coords(grid_shape_r , offset, image_shape)
            if drop_grid_proportion>0:
                mask_coords = remove_random_grid_points(mask_coords, drop_grid_proportion)
            output = get_output(batch, mask_coords, random_channel=grid_random_channel)
            if mask_X_radius>0:
                mask_coords = get_extended_mask_coordsX(mask_coords, mask_X_radius, image_shape)
            for b, c in itertools.product(range(batch.shape[0]), range(batch.shape[-1])):
                m_coords = (b,) + mask_coords + (c,)
                batch[m_coords] = constant_replacement_value_ # masking
            return output
        return fun
    elif method=="TEST":
        def fun(batch):
            image_shape = batch.shape[1:-1]
            grid_shape_ = ensure_multiplicity(len(image_shape), grid_shape)
            grid_shape_ = random_increase_grid_shape(grid_random_increase_shape, grid_shape_)
            r_patch_radius = ensure_multiplicity(len(image_shape), radius)
            if isinstance(r_patch_radius, list):
                r_patch_radius = tuple(r_patch_radius)
            mask = np.zeros(batch.shape, dtype=batch.dtype)
            mask2 = np.zeros(batch.shape, dtype=batch.dtype)
            offset = get_random_offset(grid_shape_)
            mask_coords = get_mask_coords(grid_shape_ , offset, image_shape)
            if drop_grid_proportion>0:
                mask_coords = remove_random_grid_points(mask_coords, drop_grid_proportion)
            if mask_X_radius>0:
                mask_coords = get_extended_mask_coordsX(mask_coords, mask_X_radius, image_shape)
            replacement_coords = get_random_coords(r_patch_radius, mask_coords, image_shape, exclude_X=mask_X_radius>0)
            mask_values = np.arange(mask_coords[0].shape[0])
            np.random.shuffle(mask_values)
            if grid_random_channel:
                c = np.random.randint(batch.shape[-1], size = mask_coords[0].shape[0])
                mask_coords = mask_coords + (c,)
                for b in range(batch.shape[0]):
                    m_coords = (b,) + mask_coords
                    r_coords = (b,) + replacement_coords
                    mask[m_coords] = mask_values
                    mask2[r_coords] = mask_values
                return np.concatenate([mask2, mask], axis=-1)
            else:
                for b, c in itertools.product(range(batch.shape[0]), range(batch.shape[-1])):
                    m_coords = (b,) + mask_coords + (c,)
                    r_coords = (b,) + replacement_coords + (c,)
                    mask[m_coords] = mask_values
                    mask2[r_coords] = mask_values
                return np.concatenate([mask2, mask], axis=-1)
        return fun
    else:
        raise ValueError("Invalid method")

def get_output(batch, mask_coords, random_channel=True):
    """Return the output of the blind denoising function

    Parameters
    ----------
    batch : numpy array
        format NYXC
    mask_coords : tuple of 1D-numpy arrays
        coordinates of the pixels to be masked (grid)

    Returns
    -------
    numpy array
        batch

    """
    mask = np.zeros(batch.shape, dtype=batch.dtype)
    n_pix = float(np.prod(batch.shape[1:-1])) # same number on each chan, no need to include them
    mask_value = n_pix / len(mask_coords[0])
    if random_channel and batch.shape[-1]>1:
        mask_value = mask_value * batch.shape[-1] # only grid pixel per channel
        c = np.random.randint(batch.shape[-1], size = mask_coords[0].shape[0])
        mask_coords = mask_coords + (c,)
        for b in range(batch.shape[0]):
            mask[b][mask_coords] = mask_value
    else:
        for b,c in itertools.product(range(batch.shape[0]), range(batch.shape[-1])):
            mask_idx = (b,) + mask_coords + (c,)
            mask[mask_idx] = mask_value
    return np.concatenate([batch, mask], axis=-1)

def random_increase_grid_shape(random_increase_shape, grid_shape):
    """Randomly increase the grid spacing in each direction

    Parameters
    ----------
    random_increase_shape : integer / tuple of integer
    grid_shape : tuple of integers

    Returns
    -------
    tuple of integer
        shape of the grid

    """
    shape_increase = ensure_multiplicity(len(grid_shape), random_increase_shape)
    return tuple([grid_shape[ax] + randint(0, high=shape_increase[ax]+1) if shape_increase[ax]>0 else grid_shape[ax] for ax in range(len(grid_shape))])

def get_random_offset(grid_shape):
    grid_offset = randint(0, np.product(np.array(grid_shape)))
    return np.unravel_index(grid_offset, grid_shape)

def get_mask_coords(grid_shape, offset, img_shape):
    """get coordinates of masking grid.

    Parameters
    ----------
    grid_shape : tuple of integers

    offset : tuple of integers
        offset of the grid
    img_shape : tuple

    Returns
    -------
    type
        tuple of 1D-numpy arrays corresponding to grid corrdinates for each axis

    """
    if len(offset)!=len(img_shape):
        raise ValueError("offset and shape must have same rank")
    coords = [np.arange(int(np.ceil((img_shape[i]-offset[i]) / grid_shape[i]))) * grid_shape[i] + offset[i] for i in range(len(img_shape))]
    return tuple([a.flatten() for a in np.meshgrid(*coords, sparse=False, indexing='ij')])

def remove_random_grid_points(mask_coords, drop_grid_proportion):
    if len(mask_coords[0])==1:
        return mask_coords
    n = int(drop_grid_proportion * mask_coords[0].shape[0] + 0.5)
    all_idxs = np.arange(mask_coords[0].shape[0])
    idxs_to_remove = np.random.choice(all_idxs, n, replace=False)
    idxs_to_keep = np.setdiff1d(all_idxs, idxs_to_remove)
    res = list()
    for axis in range(len(mask_coords)):
        res.append(mask_coords[axis][idxs_to_keep])
    return tuple(res)

def get_extended_mask_coordsX(mask_coords, radX, img_shape):
    """Extends mask coordinate along X-axis (useful for structured noise reduction)

    Parameters
    ----------
    mask_coords : tuple of 1D-numpy arrays
        mask coordinates
    radX : integer
        number of pixel to extend on each side of mask_coords
    img_shape : tupe
        image shape

    Returns
    -------
    tuple of 1D numpy arrays
        extended coordinates along X-axis

    """
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

def expand_coordinates(coords, radius, valid_shape, mode = "mirror"):
    radius = ensure_multiplicity(len(coords), radius)
    expand = [np.arange(-r, r+1) for r in radius]
    expanded_coords = [np.expand_dims(coords[i].reshape(-1, 1) + expand[i], axis=-1-i) for i in range(len(coords))]
    if valid_shape is not None: # out-of-bound
        assert len(valid_shape)==len(coords), "valid shape should have same size as coord"
        if mode=="mirror":
            for axis in range(len(valid_shape)):
                left = expanded_coords[axis]<0
                expanded_coords[axis][left] = - expanded_coords[axis][left]
                right = expanded_coords[axis]>=valid_shape[axis]
                expanded_coords[axis][right] = 2 * (valid_shape[axis] - 1) - expanded_coords[axis][right]
#         elif mode=="clip":
#             n_coords = coords[0].shape[0]
#             oob_coords = np.array([ ((expanded_coords[axis]<0) | (expanded_coords[axis]>=valid_shape[axis])).reshape(n_coords, -1) for axis in range(len(valid_shape))]).any(axis=0)
#             expanded_coords = [np.ma.masked_array(expanded_coords[axis], mask=np.expand_dims(oob_coords, axis=-1-axis)) for axis in range(len(expanded_coords))]
    return tuple(expanded_coords)

def get_random_coords(patch_radius, offsets, img_shape, exclude_X = False):
    """Draws random coordinates in a limited patch around points.
    Center point is excluded and drawn coordinate are ensured to be located within the image

    Parameters
    ----------
    patch_radius : integer
        radius of the patch in which random coordinate will be drawn (patch size = 2 * patch_radius + 1)
    offsets : tuple of 1D numpy arrays
        coordinates of reference points
    img_shape : tuple
        shape of the image
    exclude_X : type
        whether coordinate can be drawn along X-axis or not

    Returns
    -------
    tuple of 1D numpy arrays
        extended coordinates

    """
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

def predict_mask(model, batch, batch_mask, grid_shape):
    rank = batch.ndim - 2
    grid_shape = ensure_multiplicity(rank, grid_shape)
    n_coords = np.product(grid_shape)
    batch_masked = batch.copy()
    output = np.zeros_like(batch)
    for offset in range(n_coords):
        coords = get_mask_coords(grid_shape, offset)
        for b,c in itertools.product(range(batch.shape[0]), range(batch.shape[-1])):
            mask_idx = (b,) + mask_coords + (c,)
            batch_masked[mask_idx] = batch_mask[mask_idx]
            pred = model.predict(batch_masked)
            output[mask_idx] = pred[mask_idx]
            batch_masked[mask_idx] = batch[mask_idx]
    return output

#########################################################################################
###################### DONUT AVERAGE ####################################################


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
            raise ValueError("exclude_X incompatible with 1D arrays")
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
    """convolves batch by donut gaussian filter (center pixel excluded).

    Parameters
    ----------
    batch : numpy array
        format NYXC
    radius : integer
        radius of the gaussian filter
    exclude_X : type
        for structured noise: excludes pixels along X axis

    Returns
    -------
    type
        convolved batch

    """
    assert radius in [1,2,3], "only radius 1, 2, 3 supported"
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

def average_coord(image, extended_coords, radius =1, exclude_X=False):
    assert radius in [1,2,3], "only radius 1, 2, 3 supported"
    rank = image.ndim
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
    subarrays = image[extended_coords]
    subarrays = subarrays.reshape(subarrays.shape[0], -1)
    weights = ker.flatten()
    return np.apply_along_axis(lambda data : np.dot(data,  weights), 1, subarrays)

####################################################################################################
################################  DONUT (WEIGHTED) MEDIAN FILTER ##############################################

MEDIAN_KER_2D = {1:np.array([[1,1,1],[1,0,1],[1,1,1]])[np.newaxis,...,np.newaxis], 2:np.array([[0,0,1,0,0],[0,1,1,1,0],[1,1,0,1,1],[0,1,1,1,0],[0,0,1,0,0]])[np.newaxis,...,np.newaxis]}
MEDIAN_KER_2D_X = {1:np.array([[1,1,1],[0,0,0],[1,1,1]])[np.newaxis,...,np.newaxis], 2:np.array([[0,0,1,0,0],[0,1,1,1,0],[0,0,0,0,0],[0,1,1,1,0],[0,0,1,0,0]])[np.newaxis,...,np.newaxis]}

def median_fun(footprint):
    l = np.sum(footprint)
    idx = l//2
    footprint = footprint.flatten()!=0
    if l%2==1:
        def fun(data):
            sorted_data = np.sort(data[footprint])
            return sorted_data[idx]
        return fun
    else:
        def fun(data):
            sorted_data = np.sort(data[footprint])
            return (sorted_data[idx] + sorted_data[idx-1]) * 0.5
        return fun

def median_coord(image, extended_coords, exclude_X=False):
    subarrays = image[extended_coords]
    subarrays = subarrays.reshape(subarrays.shape[0], -1)
    size = subarrays.shape[1]
    radius = (int(np.sqrt(size))-1)//2
    assert size == (radius*2+1)**2, "invalid extended coordinates -> extension should be squared and sides uneven"
    assert radius in [1,2], "only radius 1, 2 supported"
    rank = image.ndim
    if rank==2:
        fp = MEDIAN_KER_2D[radius] if not exclude_X else MEDIAN_KER_2D_X[radius]
    else:
        raise ValueError("Only 2D arrays supported")

    return np.apply_along_axis(median_fun(fp), 1, subarrays)

def median_batch(batch, radius=1, exclude_X=False):
    assert radius in [1,2], "only radius 1, 2 supported"
    rank = batch.ndim - 2 # exclude batch & channel
    if rank==2:
        fp = MEDIAN_KER_2D[radius] if not exclude_X else MEDIAN_KER_2D_X[radius]
    else:
        raise ValueError("Only 2D arrays supported")
    return median_filter(batch, footprint = fp, mode='mirror')

def weighted_median_coord(image, extended_coords, exclude_X=False):
    subarrays = image[extended_coords]
    subarrays = subarrays.reshape(subarrays.shape[0], -1)
    size = subarrays.shape[1]
    radius = (int(np.sqrt(size))-1)//2
    assert size == (radius*2+1)**2, "invalid extended coordinates -> extension should be squared and sides uneven"
    assert radius in [1,2,3], "only radius 1, 2, 3 supported"
    rank = image.ndim
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

    weights = ker.flatten()
    return np.apply_along_axis(weighted_median_fun(weights), 1, subarrays)

def weighted_median_fun(weights):
    weights = weights.flatten()
    def fun(data):
        sorted_idxs = np.argsort(data)
        sorted_data = data[sorted_idxs]
        sorted_weights = weights[sorted_idxs]
        cs = np.cumsum(sorted_weights)
        p = cs-0.5*sorted_weights # weights sum to 1: no need to divide by cs[-1]
        return np.interp(0.5, p, sorted_data)
    return fun

def weighted_median_coord(image, extended_coords, radius =1, exclude_X=False):
    assert radius in [1,2,3], "only radius 1, 2, 3 supported"
    rank = image.ndim
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
    subarrays = image[extended_coords]
    subarrays = subarrays.reshape(subarrays.shape[0], -1)
    weights = ker.flatten()
    return np.apply_along_axis(weighted_median_fun(weights), 1, subarrays)

def weighted_median_batch(batch, radius=1, exclude_X=False):
    assert radius in [1,2,3], "only radius 1, 2, 3 supported"
    #return median_filter(batch, footprint = MEDIAN_KER_2D_X[radius-1] if exclude_X else MEDIAN_KER_2D[radius-1], mode='mirror')
    rank = batch.ndim - 2 # exclude batch & channel
    size = 2*radius+1
    if rank==2:
        ker = AVG_KERNELS_2D[radius] if not exclude_X else AVG_KERNELS_2D_X[radius]
        fp = np.ones((size, size))[np.newaxis,...,np.newaxis]
    elif rank==3:
        ker = AVG_KERNELS_3D[radius] if not exclude_X else AVG_KERNELS_3D_X[radius]
        fp = np.ones((size, size, size))[np.newaxis,...,np.newaxis]
    elif rank==1:
        if exclude_X:
            raise ValueError("exclude_X incopatible with 1D arrays")
        ker = AVG_KERNELS_1D[radius]
        fp = np.ones((size))[np.newaxis,...,np.newaxis]
    else:
        raise ValueError("Only 1D, 2D or 3D arrays supported")
    return generic_filter(batch, weighted_median_fun(ker), footprint=fp)
    #return generic_filter(batch, lambda data :weighted_median(data, ker.flatten()), footprint=fp)


#############################################################################################################################################
# color denoising -> adapted from: https://github.com/NVlabs/selfsupervised-denoising/blob/master/selfsupervised_denoising.py
# NOT TESTED YET
import tensorflow as tf

def batch_vtmv(v, m): # Batched (v^T * M * v).
    return tf.reduce_sum(v[..., :, tf.newaxis] * v[..., tf.newaxis, :] * m, axis=[-2, -1])

def computeStS(sigma_x):
    # Calculate A^T * A
    c00 = sigma_x[..., 0]**2 + sigma_x[..., 1]**2 + sigma_x[..., 2]**2 # NHW
    c01 = sigma_x[..., 1]*sigma_x[..., 3] + sigma_x[..., 2]*sigma_x[..., 4]
    c02 = sigma_x[..., 2]*sigma_x[..., 5]
    c11 = sigma_x[..., 3]**2 + sigma_x[..., 4]**2
    c12 = sigma_x[..., 4]*sigma_x[..., 5]
    c22 = sigma_x[..., 5]**2
    c0 = tf.stack([c00, c01, c02], axis=-1) # NHW3
    c1 = tf.stack([c01, c11, c12], axis=-1) # NHW3
    c2 = tf.stack([c02, c12, c22], axis=-1) # NHW3
    return tf.stack([c0, c1, c2], axis=-1) # NHW33

def color_gauss_loss(y_true, y_pred, sigma_x, sigma2_n, regularization=False, dtype=tf.float32):
    """loss for normal distribution of color image.
    Adapted from Laine et al. 2019
    https://github.com/NVlabs/selfsupervised-denoising/blob/master/selfsupervised_denoising.py

    Parameters
    ----------
    y_true : tensor NHW3
        noisy input
    y_pred : tensor NHW3
        predicted clean values
    sigma_x : tensor NHW6
        std predictions (for each color channel as well as covariances)
    sigma2_n : tensor NHW3
        std of noise for each color channel
    regularization : bool
        loss regularization
    dtype : tensorflow datatype

    Returns
    -------
    loss tensor NHW

    """
    I = tf.eye(3, batch_shape=[1, 1, 1], dtype=dtype)
    sigma2_n_ = sigma2_n[..., tf.newaxis] * I # NHWC1 * NHWCC = NHWCC
    sigma_y = computeStS(sigma_x) + sigma2_n_ # NHWCC, total covariance matrix. Cannot be singular because sigma_n is at least a small diagonal.
    sigma_y_inv = tf.linalg.inv(sigma_y) # NHWCC
    l2 = batch_vtmv(y_true - y_pred, sigma_y_inv) # NHW
    dets = tf.linalg.det(sigma_y) # NHW
    #dets = tf.maximum(zero64, dets) # NHW. Avoid division by zero and negative square roots. # we predict exp(sigma) so no problem
    loss = 0.5 * ( l2 + K.log(dets) ) # NHW
    if regularization:
        loss = loss - 0.1 * tf.reduce_mean(sigma2_n, axis=-1) # Balance regularization.
    return loss

#########################################################################################################
##############"" MISC ########### FOR TESTING PURPOSES
def get_signal_frequency_balanced_mask(batch, remove_proportion, probability_fun):
    mask = np.zeros_like(batch)
    for b, c in itertools.product(range(batch.shape[0]), range(batch.shape[-1])):
        im_flat = batch[b,...,c].reshape(-1)
        n = int(remove_proportion * im_flat.shape[0] + 0.5)
        all_idxs = np.arange(im_flat.shape[0])
        idxs_to_remove = np.random.choice(all_idxs, size=n, replace=False, p=probability_fun(im_flat))
        idx_keep = np.setdiff1d(all_idxs, idxs_to_remove)
        mask_image = mask[b,...,c].reshape(-1)
        mask_image[idx_keep] = im_flat.shape[0] / n
    return mask

def get_proba_fun(histogram, breaks): # COULD BE OPTIMIZED -> COMPUTE BIN USING MIN/MAX/NBIN instead of digitize
    breaks = breaks[1:]
    def proba_fun(values):
        bin_idx = np.digitize(values, breaks, right=False)
        probas = histogram[bin_idx]
        return probas / sum(probas)
    return proba_fun

########################################################################################
######################## UTILS #########################################################
def extract_along_batch_and_channel(batch, coords):
    output = np.zeros(shape=(batch.shape[0], batch.shape[-1], coords[0].shape[0]))
    for b, c in itertools.product(range(batch.shape[0]), range(batch.shape[-1])):
        output[b, c] = batch[b,...,c][coords]
    return output
