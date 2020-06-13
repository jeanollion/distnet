import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import losses
import numpy as np
from .lovasz_losses_tf import lovasz_hinge, lovasz_softmax
from .helpers import convert_probabilities_to_logits, ensure_multiplicity

def mse(y_true, y_pred):
    return K.square(y_pred - y_true)

def mae(y_true, y_pred):
    return K.abs(y_pred - y_true)

def loss_laplace(epsilon = 1e-6):
    def loss_fun(y_true, y_pred):
        n_chan = K.shape(y_true)[-1]
        mu = y_pred[...,:n_chan]
        sigma = y_pred[...,n_chan:]
        if epsilon>0:
            sigma+=epsilon
        return K.abs( ( mu - y_true ) / sigma ) + K.log( sigma )
    return loss_fun

def loss_gauss(epsilon = 1e-6):
    def loss_fun(y_true, y_pred):
        n_chan = K.shape(y_true)[-1]
        mu = y_pred[...,:n_chan]
        sigma2 = y_pred[...,n_chan:]
        if epsilon>0:
            sigma2+=epsilon
        return 0.5 * ( K.square( mu - y_true ) / sigma2 + K.log( sigma2 ) )
    return loss_fun

def mother_machine_mask(shape=(256, 32), mask_size=40, lower_end_only=True, dtype=np.float32):
    mask = np.ones(shape=(1,)+shape, dtype=dtype) #+(1,)
    val = lambda y : ((shape[0] -1 - y )/mask_size)**3
    for y in range(shape[0]-mask_size, shape[0]):
        mask[:,y]=val(y)
    if not lower_end_only:
        for y in range(0, mask_size):
            mask[:,y]=val(shape[0]-1-y)
    return mask

def masked_loss(original_loss_func, mask):
    mask_tf = tf.convert_to_tensor(mask)
    def loss_func(true, pred):
        loss = original_loss_func(true, pred)
        loss = loss * mask #broadcasting to batch size , y, x , channels
        return loss
    return loss_func

def ssim_loss(max_val = 1, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
    def loss_fun(y_true, y_pred):
        SSIM = tf.image.ssim(y_true, y_pred, max_val=max_val, filter_size=filter_size, filter_sigma=filter_sigma, k1=k1, k2=k2)
        return 1 - (1 + SSIM ) * 0.5
    return loss_fun

def mix_losses(losses, weights, reshape_axis_list=None):
    assert len(losses)==len(weights), "Weigh_ts array should be of same length as loss array"
    reshape_axis_list = ensure_multiplicity(len(losses), reshape_axis_list)
    def loss_func(y_true, y_pred):
        for i in range(0, len(losses)):
            loss = losses[i](y_true, y_pred) * weights[i]
            if reshape_axis_list[i] is not None:
                loss = K.reshape(loss, reshape_axis_list[i])
            res = loss if i==0 else res + loss
        return res
    return loss_func

def weighted_loss_by_category(original_loss_func, weights_list, axis=-1, sparse=True):
    def loss_func(true, pred):
        if sparse:
            class_selectors = K.squeeze(true, axis=axis)
        else:
            class_selectors = K.argmax(true, axis=axis)

        #considering weights are ordered by class, for each class
        #true(1) if the class index is equal to the weight index
        class_selectors = [K.equal(float(i), class_selectors) for i in range(len(weights_list))]

        #casting boolean to float for calculations
        #each tensor in the list contains 1 where ground true class is equal to its index
        #if you sum all these, you will get a tensor full of ones.
        class_selectors = [K.cast(x, K.floatx()) for x in class_selectors]

        #for each of the selections above, multiply their respective weight
        weights = [sel * w for sel, w in zip(class_selectors, weights_list)]

        #sums all the selections
        #result is a tensor with the respective weight for each element in predictions
        weight_multiplier = weights[0]
        for i in range(1, len(weights)):
            weight_multiplier = weight_multiplier + weights[i]

        #make sure your original_loss_func only collapses the class axis
        #you need the other axes intact to multiply the weights tensor
        loss = original_loss_func(true, pred)
        loss = loss * weight_multiplier
        return loss
    return loss_func

def pixelwise_weighted_loss(original_loss_func, y_true_channels=None, weight_channels=None, sum_channels=True, reshape_axis=None):
    '''
    This function implements pixel-wise weighted loss
    if y_true has 2n channels, weight maps are the Weight map are [n, 2n) channels; otherwise y_true channels and weight channels can be specified as lists of length 2
    '''
    #@tf.function
    if y_true_channels is None:
        def loss_func(y_true, y_pred):
            y_true, weightMap = tf.split(y_true, 2, axis=-1)
            loss = original_loss_func(y_true, y_pred)
            if reshape_axis is not None:
                loss = K.reshape(loss, reshape_axis)
            loss = loss * weightMap
            if sum_channels:
                return K.sum(loss, -1)
            else:
                return loss
    else:
        def loss_func(y_true, y_pred):
            weightMap = y_true[...,weight_channels[0]:weight_channels[1]]
            y_true = y_true[...,y_true_channels[0]:y_true_channels[1]]
            loss = original_loss_func(y_true, y_pred)
            if reshape_axis is not None:
                loss = K.reshape(loss, reshape_axis)
            loss = loss * weightMap
            if sum_channels:
                return K.sum(loss, -1)
            else:
                return loss
    return loss_func

def binary_dice_coeff(y_true, y_pred, smooth=1e-10, batch_mean=False, square_norm=False):
    batchSize = K.shape(y_true)[0]
    t = K.reshape(y_true, shape=(batchSize, -1))
    p = K.reshape(y_pred, shape=(batchSize, -1))
    tp = K.sum(t * p, -1)
    tv = K.sum(t, -1)
    pv = K.sum(p, -1)
    if batch_mean:
        tv = K.mean(tv, 0, keepdims=True)
        pv = K.mean(pv, 0, keepdims=True)
    if square_norm:
        tv = K.square(tv)
        pv = K.square(pv)
    return (tp + smooth) / ( 0.5 * (tv + pv) + smooth)

def binary_dice_loss(smooth=1e-10, batch_mean=False, square_norm=False):
    return lambda y_true, y_pred : 1 - binary_dice_coeff(y_true, y_pred, smooth, batch_mean, square_norm)

def soft_dice_loss(smooth=1e-10, batch_mean=False, square_norm=False, sparse=True):
    def loss_fun(y_true, y_pred):
        if not sparse:
            y_true = y_true[...,-1:]
        return binary_dice_loss(smooth=smooth, batch_mean=batch_mean, square_norm=square_norm)(y_true, y_pred[...,-1:])
    return loss_fun

def binary_generalized_dice_loss(batch_mean=False, square_weights = True):
    def loss_fun(y_true, y_pred):
        tshape = K.shape(y_true)
        batchSize = tshape[0]
        t = K.reshape(y_true, shape=(batchSize, -1))
        p = K.reshape(y_pred, shape=(batchSize, -1))
        tb = 1 - t
        pb = 1 - p
        tp = K.sum(t * p, -1)
        tn = K.sum(tb * pb, -1)
        tv = K.sum(t, -1)
        pv = K.sum(p, -1)
        tbv = K.sum(tb, -1)
        pbv = K.sum(pb, -1)
        if batch_mean:
            tv = K.mean(tv, 0, keepdims=True)
            pv = K.mean(pv, 0, keepdims=True)
            tbv = K.mean(tbv, 0, keepdims=True)
            pbv = K.mean(pbv, 0, keepdims=True)
        if square_weights:
            w = 1. / K.square(tv)
            wb = 1. / K.square(tbv)
        else:
            w = 1. / tv
            wb = 1. / tbv
        w = tf.where(tf.math.is_inf(w), K.ones_like(w), w) # regularize 0 div
        wb = tf.where(tf.math.is_inf(wb), K.ones_like(wb), wb)
        return 1 - ( w * tp + wb * tn ) / ( 0.5 * w * ( tv + pv ) + 0.5 * wb * ( tbv + pbv ) )
    return loss_fun

def soft_generalized_dice_loss(batch_mean = False, square_weights = True, sparse = True, exclude_background=False, spatial_dim_axes=[1, 2]):
    """ Generalised Dice Loss function defined in
        Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
        loss function for highly unbalanced segmentations. DLMIA 2017

        TF1x implementation : https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py
        Assumes class axis = -1

    Parameters
    ----------
    batch_mean : type
        if true loss is computed for the whole batch else for each image of the batch
    square_weights : bool
        if true class weight is inverse squared volume of the class else inverse volume
    sparse : boolean
        wheter y_true is sparse

    Returns
    -------
    type
        loss function

    """
    def loss_fun(y_true, y_pred):
        if sparse:
            y_true = K.cast(y_true[...,0], "int32")
            y_true = K.one_hot(y_true, K.shape(y_pred)[-1])
        if exclude_background:
            y_true = y_true[...,1:]
            y_pred = y_pred[...,1:]
        inter = K.sum(y_true * y_pred, spatial_dim_axes)
        tv = K.sum(y_true, spatial_dim_axes)
        pv = K.sum(y_true, spatial_dim_axes)
        if batch_mean:
            tv = K.mean(tv, 0, keepdims=True)
            pv = K.mean(pv, 0, keepdims=True)
        if square_weights:
            w = 1. / K.square(tv)
        else:
            w = 1. / tv
        w = tf.where(tf.math.is_inf(w), K.ones_like(w), w) # regularize 0 div by zero
        return 1 - (K.sum(w * inter, -1) + 1e-10) / (K.sum(w * 0.5 * (tv + pv), -1) + 1e-10) # sum for each class
    return loss_fun

def binary_tversky_loss(alpha=0.3, beta=0.7, smooth=1e-7, batch_mean=False):
    """Return the Tversky loss for imbalanced data
        Sadegh et al. (2017)
        Tversky loss function for image segmentation using 3D fully convolutional deep networks
    Parameters
    ----------
    alpha : type
        weight of false positives (penalize false positives)
    beta : type
        weight of false negatives (penalize false negatives)
    smooth : type
        Description of parameter `smooth`.
    batch_mean : type
        sum over batch dimension

    Returns
    -------
    type
        loss function

    """
    def loss_fun(y_true, y_pred):
        batchSize = K.shape(y_true)[0]
        t = K.reshape(y_true, shape=(batchSize, -1))
        p = K.reshape(y_pred, shape=(batchSize, -1))

        tp = K.sum(t * p, -1)
        fp = K.sum((1 - t) * p, -1)
        fn = K.sum(t * (1 - p), -1)
        if batch_mean:
            fp = K.mean(fp, 0, keepdims=True)
            fn = K.mean(fn, 0, keepdims=True)
            tpm = K.mean(tp, 0, keepdims=True)
        else:
            tpm = tp
        return 1 - (tp + smooth) / (tpm + alpha * fp + beta * fn + smooth)
    return loss_fun

def boundary_regional_loss(alpha, regional_loss, mul_coeff=1, y_true_channels=None, levelset_channels=None):
    """Mixed boundary loss with regional loss function as in  https://arxiv.org/abs/1812.07032

    Parameters
    ----------
    alpha : type
        number / Keras variable in range [0,1] importance given to regional loss over boundary loss
    regional_loss : function. returns a tensor with shape (batch_size, )
    mul_coeff : multiplicative coefficient applied to the boundary loss to balance with regional loss
    Returns
    -------
    type
        loss function that inputs:
        y_true : type
            ground truth tensor, concatenated with level sel (distance map from bounds, negative inside and positive outside)
        y_pred : type
            predicted tensor
    PyTorch implementation : https://github.com/LIVIAETS/surface-loss/blob/master/losses.py
    """
    def loss_fun(y_true, y_pred):
        if y_true_channels is None:
            channels = K.shape(y_true)[-1]
            mid = channels // 2
            levelset = y_true[...,mid:]
            y_true = y_true[...,0:mid]
        else:
            levelset = y_true[...,levelset_channels[0]:levelset_channels[1]]
            y_true = y_true[...,y_true_channels[0]:y_true_channels[1]]

        rl = regional_loss(y_true, y_pred)
        bl = K.sum(levelset * y_pred, [1, 2, 3])
        if mul_coeff!=1:
            bl = bl * mul_coeff
        return  alpha * rl + (1 - alpha) * bl
    return loss_fun

def binary_lovasz_loss(from_logits = False, per_image = False, ignore=None):
    def loss_fun(yt, yp):
        if not from_logits:
            yp = convert_probabilities_to_logits(yp)
        return lovasz_hinge(logits=yp, labels=yt, per_image=per_image, ignore=ignore)
    return loss_fun

def soft_lovasz_loss(sparse=True, per_image=False, classes='present', ignore=None):
    def loss_fun(yt, yp):
        if not sparse:
            yt = K.argmax(yt, -1)
        else :
            yt = K.squeeze(yt, -1)
        return lovasz_softmax(probas=yp, labels=yt, per_image=per_image, classes=classes, ignore=ignore)
    return loss_fun
# segmetation loss keras: https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py
#focal loss : pip install focal-loss
#from focal_loss import binary_focal_loss
#binary_focal_loss(true, pred, gamma, pos_weight=alpha/(1-alpha)) * (1-alpha). # to get the focal loss with default parameters
