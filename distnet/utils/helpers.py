import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np
import shutil

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
    if overwrite:
        try:
            shutil.rmtree(outdir)
        except:
            pass
    tf.saved_model.simple_save(K.get_session(), export_dir=outdir, inputs=inputs, outputs=outputs)
    print("inputs: {}, outputs: {}".format(inputs, outputs))

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
