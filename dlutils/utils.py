import h5py
import keras.backend as K
import tensorflow as tf

def h5py_dataset_iterator(g, prefix=''):
    for key in g.keys():
        item = g[key]
        path = '{}/{}'.format(prefix, key)
        if isinstance(item, h5py.Dataset): # test for dataset
            yield (path, item)
        elif isinstance(item, h5py.Group): # test for group (go down)
            yield from h5py_dataset_iterator(item, path)

def get_datasets_paths(h5py_file, suffix, group_keyword=None):
    return [path for (path, ds) in h5py_dataset_iterator(h5py_file) if path.endswith(suffix) and (group_keyword==None or group_keyword in path)]

def get_datasets_by_path(h5py_file, paths):
    return [h5py_file[path] for path in paths]

def get_datasets(h5py_file, suffix, group_keyword=None):
    return [ds for (path, ds) in h5py_dataset_iterator(h5py_file) if path.endswith(suffix) and (group_keyword==None or group_keyword in path)]

def get_parent_path(path):
    idx = path.rfind('/')
    if idx>0:
        return path[:idx]
    else:
        return None

def get_earse_small_values_function(thld):
    def earse_small_values(im):
        im[im<thld]=0
        return im
    return earse_small_values



def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
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
def export_model_graph(model, outdir, filename="saved_model.pb", output_names=None):
    if output_names:
        if len(output_names)!=len(model.outputs):
            raise ValueError("Model has {} outputs whereas {} output names are specified".format(len(model.outputs), len(output_names)))
        for out in model.outputs:
            tf.identity(out, name=output_names[i])
    frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])
    tf.train.write_graph(frozen_graph, outdir, filename, as_text=False)

def export_model_bundle(model, outdir):
    outputs = dict(zip([out.op.name for out in model.outputs], model.outputs))
    inputs = dict(zip([in.op.name for in in model.inputs], model.inputs))
    tf.saved_model.simple_save(K.get_session(), export_dir=outdir, inputs=inputs, outputs=outputs)
    #print("inputs: {}, outputs: {}".format(inputs, outputs))
