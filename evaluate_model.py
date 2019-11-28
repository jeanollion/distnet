import sys
sys.path.append('/data/Dev/DL_Utils')
import dlutils.keras_models as km
from dlutils.utils import evaluate_model
from keras.optimizers import Adam
from keras import metrics
import itertools

# create model
n_stack = 4
model = km.get_unet_model(image_shape=(256,32), filters=48, n_down=4, n_input_channels=2, n_outputs=3, n_output_channels=[1, 4, 2], out_activations=["linear", "softmax", "linear"], n_stack=n_stack, n_1x1_conv=1, stacked_intermediate_outputs=False, stacked_skip_conection=True, double_n_filters=True)
model.compile(optimizer=Adam(2e-5),
              loss=["mean_absolute_error", 'sparse_categorical_crossentropy', 'mean_squared_error']*n_stack,
              loss_weights=[0.5, 1, 1]*n_stack,
              metrics={"output1":metrics.sparse_categorical_accuracy}
             )
model.load_weights("/data/Images/Phase/DL_models/seg_track_48x4_cpL.h5")

# create iterator
params = dict(h5py_file_path='/data/Images/MOP/data_segDis/bacteriaSegDis.h5',
             channel_keywords=['/raw', '/regionLabels', '/prevRegionLabels', '/edm'],
             image_data_generators=None,
             channels_prev=[True, True, False, True],
             input_channels=[0],
             output_channels=[1, 2, 3],
             return_categories = True,
             erase_cut_cell_length = 30,
             mask_channels=[1, 2, 3],
             perform_data_augmentation=False,
             shuffle=False)
train_it = H5dyIterator(group_keyword="ds_noError", **params)
test_it = H5dyIterator(group_keyword="ds_noError_test", **params)


losses_names = ["Loss0", "Loss1", "Loss2"]
losses_names = ["GlobalLoss"] + [l+"_"+str(i) for i, l in itertools.product(range(1, 4), losses_names)] + losses_names

evaluate_model(iterator, model, losses_names=losses_names, acc_names=["category_accuracy"])
