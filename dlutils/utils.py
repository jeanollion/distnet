import h5py

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
