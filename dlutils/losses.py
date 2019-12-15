import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import losses
import numpy as np

def categorical_focal_loss(gamma=2., alpha=.25, sparse=True):
    """
    From https://github.com/umbertogriffo/focal-loss-keras
    Softmax version of focal loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        if sparse:
            cross_entropy = losses.sparse_categorical_crossentropy(y_true, y_pred)
        else:
            cross_entropy = losses.categorical_crossentropy(y_true, y_pred)
        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        return loss
        # Sum the losses in mini_batch
        #return K.sum(loss, axis=1)

    return categorical_focal_loss_fixed

def mother_machine_mask(shape=(256, 32), mask_size=40, lower_end_only=True):
    mask = np.ones(shape=(1,)+shape, dtype=np.float) #+(1,)
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

def ssim_loss(y_true, y_pred):
  SSIM = tf.image.ssim(y_true, y_pred, max_val=256)
  return 1 - (1 + SSIM ) /2

def sum_losses(losses, weights):
    if len(losses)!=len(weights):
        raise ValueError("Weights array should be of same length as loss array")
    def loss_func(y_true, y_pred):
        loss_values = [loss(y_true, y_pred) * w for loss, w in zip(losses, weights)]
        return sum(loss_values)

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

def pixelwise_weighted_loss(original_loss_func):
    '''
    This function implements pixel-wise weighted loss
    if y_true has 2n channels, weight maps are the Weight map are [n, 2n) channels
    '''
    def loss_func(y_true, y_pred):
        channels = tf.shape(y_true)[-1]
        #if channels%2!=0:
        #    raise ValueError("invalid channel number for y_true: should be a multiple of 2")
        mid = channels // 2

        weightMap = y_true[...,mid:]
        y_true = y_true[...,0:mid]
        
        loss = original_loss_func(y_true, y_pred)
        loss = tf.expand_dims(loss, -1)
        return loss * weightMap
        #return K.mean(loss * weightMap, axis=-1)
    return loss_func
