import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, LearningRateScheduler
import numpy as np
import shutil


##
## define the callback :
#set the loss function:
#add the callback  to the model
#remember to set pre_processing.level_set as weightmap function

class EpochNumberCallback(Callback):
    """Callback that allows to have a keras variable that depends on epoch number
    Useful to mix 2 losses with relative weights that vary with epoch number as in https://arxiv.org/abs/1812.07032
    Use case with boundary loss:
    alpha_cb = EpochNumberCallback(EpochNumberCallback.linear_decay(n_epochs, 0.01))
    loss_fun = boundary_regional_loss(alpha_cb.get_variable(), regional_loss_fun)

    Parameters
    ----------
    fun : function (int -> float)
        function applied to epoch number

    Attributes
    ----------
    variable : keras variable
        variable updated by fun at each epoch
    fun : function
        Function

    """
    def __init__(self, fun=lambda v : v ):
        self.fun = fun
        self.variable = K.variable(fun(0))

    def on_epoch_end(self, epoch, logs={}):
        K.set_value(self.variable, self.fun(epoch + 1))

    def get_variable(self):
        return self.variable

    @staticmethod
    def linear_decay(total_epochs, minimal_value):
        assert minimal_value<=1, "minimal value must be <=1"
        return lambda current_epoch : max(minimal_value, 1 - current_epoch / total_epochs)

    @staticmethod
    def switch(epoch, before=1., after=0):
        return lambda current_epoch : before if current_epoch<epoch else after

    @staticmethod
    def soft_switch(epoch_start, epoch_end, before=1., after=0):
        assert epoch_start<epoch_end
        def fun(current_epoch):
            if current_epoch<=epoch_start:
                return before
            elif current_epoch>=epoch_end:
                return after
            else:
                alpha  = (current_epoch - epoch_start) / (epoch_end - epoch_start)
                return before * (1 - alpha) + after * alpha
        return fun

def convert_probabilities_to_logits(y_pred): # y_pred should be a tensor: tf.convert_to_tensor(y_pred, np.float32)
      y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
      return K.log(y_pred / (1 - y_pred))

def ensure_multiplicity(n, object):
    if object is None:
        return [None] * n
    if not isinstance(object, (list, tuple)):
        object = [object]
    if len(object)>1 and len(object)!=n:
        raise ValueError("length should be either 1 either {}".format(n))
    if n>1 and len(object)==1:
        object = object*n
    elif n==0:
        return []
    return object

def flatten_list(l):
    flat_list = []
    for item in l:
        append_to_list(flat_list, item)
    return flat_list

def append_to_list(l, element):
    if isinstance(element, list):
        l.extend(element)
    else:
        l.append(element)

def get_earse_small_values_function(thld):
    def earse_small_values(im):
        im[im<thld]=0
        return im
    return earse_small_values

def step_decay_schedule(initial_lr=1e-3, minimal_lr=1e-5, decay_factor=0.50, step_size=50):
    if minimal_lr>initial_lr:
        raise ValueError("Minimal LR should be inferior to initial LR")
    def schedule(epoch):
        lr = max(initial_lr * (decay_factor ** np.floor(epoch/step_size)), minimal_lr)
        return lr
    return LearningRateScheduler(schedule, verbose=1)

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen, or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

# caution: export with a version of tensorflow compatible with the version that will run the prediction. e.g export with 1.14 is not compatible with 1.13.1
def export_model_graph(model, outdir, filename="saved_model.pb", input_names=None, output_names=None):
    if input_names:
        if len(input_names)!=len(model.inputs):
            raise ValueError("Model has {} inputs whereas {} input names are specified".format(len(model.outputs), len(output_names)))
        for i, input in enumerate(model.inputs):
            input._name = input_names[i]
            #tf.identity(input, name=input_names[i])
    if output_names:
        if len(output_names)!=len(model.outputs):
            raise ValueError("Model has {} outputs whereas {} output names are specified".format(len(model.outputs), len(output_names)))
        for i, out in enumerate(model.outputs):
            out._name = output_names[i]
            #tf.identity(out, name=output_names[i])
    for iname in model.inputs:
        print("input", iname._name)
    frozen_graph = freeze_session(K.get_session(), output_names=output_names)
    tf.train.write_graph(frozen_graph, outdir, filename, as_text=False)

def export_model_bundle(model, outdir, overwrite=False):
    outputs = dict(zip([out.op.name for out in model.outputs], model.outputs))
    inputs = dict(zip([input.op.name for input in model.inputs], model.inputs))
    print("inputs: {}, outputs: {}".format(inputs, outputs))
    if overwrite:
        try:
            shutil.rmtree(outdir)
        except:
            pass
    tf.saved_model.simple_save(K.get_session(), export_dir=outdir, inputs=inputs, outputs=outputs) # versions 1.13 - 1.15
    #tf.saved_model.save(model, export_dir=outdir) # version >=1.15


def evaluate_model(iterator, model, metrics, metric_names, xp_idx_in_path=2, position_idx_in_path=3, progress_callback=None):
    try:
        import pandas as pd
    except ImportError as error:
        print("Pandas not installed")
        return

    arr, paths, labels, indices = iterator.evaluate(model, metrics, progress_callback=progress_callback)
    df = pd.DataFrame(arr)
    if len(metric_names)+2 != df.shape[1]:
        raise ValueError("Invalid loss / accuracy name: expected: {} names, got: {} names".format(df.shape[1]-2, len(metric_names)))
    df.columns=["Idx", "dsIdx"]+metric_names
    df["Indices"] = pd.Series(indices)
    dsInfo = np.asarray([p.split('/') for p in paths])
    df["XP"] = pd.Series(dsInfo[:,xp_idx_in_path])
    df["Position"] = pd.Series(dsInfo[:,position_idx_in_path])
    return df

def displayProgressBar(max): # this progress bar is compatible with google colab
    from IPython.display import HTML, display
    def progress(value=0, max=max):
        return HTML("""
            <progress
                value='{value}'
                max='{max}',
                style='width: 100%'
            >
                {value}
            </progress>
        """.format(value=value, max=max))
    out = display(progress(), display_id=True)
    currentProgress=[0]
    def callback():
        currentProgress[0]+=1
        out.update(progress(currentProgress[0]))
    return callback

def predict_average_flip_rotate(model, batch, rotate90 = True, list_flips=[0,1,2]):
    if not isinstance(list_flips, (tuple, list)):
        list_flips = [list_flips]
    batch_list = _append_flip_and_rotate_list(batch, list_flips, rotate90)
    predicted_list = [model(b) for b in batch_list]
    # transform back
    if isinstance(predicted_list[0], (tuple, list)):
        predicted_list = _transpose(predicted_list)
        return tuple([_reverse_and_mean(l, rotate90, list_flips) for l in predicted_list])
    else:
        return _reverse_and_mean(predicted_list, rotate90, list_flips)

def _append_flip_and_rotate_list(batch, rotate90=True, list_flips=[0,1,2]):
    if isinstance(batch, (tuple, list)):
        batch_list = []
        for i in range(len(batch)):
            batch_list.append(_append_flip_and_rotate(batch, list_flips, rotate90))
        return _transpose(batch_list)
    else:
        return _append_flip_and_rotate(batch, list_flips, rotate90)

def _append_flip_and_rotate(batch, rotate90=True, list_flips=[0,1,2]):
    trans = [batch] + [AUG_FUN_2D[flip+1](batch) for flip in list_flips]
    if rotate90:
        trans +=[AUG_FUN_2D[4](batch)]
        trans = trans + [AUG_FUN_2D[i+5](batch) for i in list_flips]
    return trans

def _reverse_and_mean(image_list, rotate90 = True, list_flips=[0,1,2]):
    n_flips = len(list_flips)
    for idx, i in enumerate(list_flips):
        image_list[idx+1] = AUG_FUN_REV_2D[i+1](image_list[idx+1])
    if rotate90:
        image_list[n_flips+1] = AUG_FUN_REV_2D[4](image_list[n_flips+1])
        for idx, i in enumerate(list_flips):
            img_idx = idx + n_flips + 2
            image_list[img_idx] = AUG_FUN_REV_2D[i + 5](image_list[img_idx])
    return np.mean(image_list, axis=0)

def _transpose(list_of_list):
    size1=len(list_of_list)
    size2=len(list_of_list[0])
    return [ [ list_of_list[i][j] for i in range(size1)] for j in range(size2) ]

AUG_FUN_2D = [
    lambda img : img,
    lambda img : np.flip(img, axis=1),
    lambda img : np.flip(img, axis=2),
    lambda img : np.flip(img, axis=(1, 2)),
    lambda img : np.rot90(img, k=1, axes=(1,2)),
    lambda img : np.rot90(img, k=3, axes=(1,2)), # rot + flip0
    lambda img : np.rot90(np.flip(img, axis=2), k=1, axes=(1,2)),
    lambda img : np.rot90(np.flip(img, axis=(1, 2)), k=1, axes=(1,2))
]
AUG_FUN_REV_2D = [
    lambda img : img,
    lambda img : np.flip(img, axis=1),
    lambda img : np.flip(img, axis=2),
    lambda img : np.flip(img, axis=(1, 2)),
    lambda img : np.rot90(img, k=3, axes=(1,2)),
    lambda img : np.rot90(img, k=1, axes=(1,2)), # rot + flip0
    lambda img : np.rot90(np.flip(img, axis=2), k=1, axes=(1,2)),
    lambda img : np.rot90(np.flip(img, axis=(1, 2)), k=3, axes=(1,2))
]

def get_nd_gaussian_kernel(radius=1, sigma=0, ndim=2):
    size = 2 * radius + 1
    if ndim == 1:
        coords = [np.mgrid[-radius:radius:complex(0, size)]]
    elif ndim==2:
        coords = np.mgrid[-radius:radius:complex(0, size), -radius:radius:complex(0, size)]
    elif ndim==3:
        coords = np.mgrid[-radius:radius:complex(0, size), -radius:radius:complex(0, size), -radius:radius:complex(0, size)]
    elif ndim==4:
        coords = np.mgrid[-radius:radius:complex(0, size), -radius:radius:complex(0, size), -radius:radius:complex(0, size), -radius:radius:complex(0, size)]
    else:
        raise ValueError("Up to 4D supported")

    # Need an (N, ndim) array of coords pairs.
    stacked = np.column_stack([c.flat for c in coords])
    mu = np.array([0.0]*ndim)
    s = np.array([sigma if sigma>0 else radius]*ndim)
    covariance = np.diag(s**2)
    z = multivariate_normal.pdf(stacked, mean=mu, cov=covariance)
    z = z.reshape(coords[0].shape) # Reshape back to a (N, N) grid.
    return z/z.sum()
