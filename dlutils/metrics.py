import numpy as np

def sparse_categorical_accuracy(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis=-1)
    y_true = y_true.squeeze(-1)
    return np.equal(y_true, y_pred, dtype=np.float)

def category_mask(classes):
    if np.isscalar(classes):
        classes = [classes]
    def mask_fun(y_true):
        weights = np.zeros(shape=y_true.shape)
        for c in classes:
            weights[y_true==c]=1
        return weights
    return mask_fun

def metric_mask(metric, mask_fun = lambda y_true:np.abs(y_true)>1e-5, null_mask_return_value=1):
    def metric_m(y_true, y_pred):
        mask = mask_fun(y_true)
        metric_value = metric(y_true, y_pred)
        if len(metric_value.shape)==len(mask.shape)-1:
            mask = mask.squeeze(-1)
        avg_axis = tuple(range(1,len(metric_value.shape)))
        try :
            return np.average(metric_value, axis=avg_axis, weights=mask)
        except ZeroDivisionError:
            return null_mask_return_value
    return metric_m

def sparse_categorical_accuracy_mask(classes, null_mask_return_value=1):
    return metric_mask(metric=sparse_categorical_accuracy, mask_fun = category_mask(classes), null_mask_return_value=null_mask_return_value)

def mae(y_true, y_pred):
    return np.abs(y_true-y_pred)

def mse(y_true, y_pred):
    return np.square(y_true-y_pred)

def r_square(y_true, y_pred):
    avg_axis = tuple(range(1,len(y_true.shape)))
    SS_res =  np.sum(np.square(y_true - y_pred), axis=avg_axis)
    SS_tot = np.sum(np.square(y_true - np.mean(y_true, axis=avg_axis, keepdims=True)), axis=avg_axis)
    res =  (1 - SS_res/(SS_tot + np.finfo(np.float32).eps))
    return res
