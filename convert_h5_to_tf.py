import sys
sys.path.append('/data/Dev/DL_Utils')
import dlutils.keras_models as km
from dlutils.utils import export_model_graph, export_model_bundle
import shutil
#model_edm, model_dy, model_dy_predict = get_edm_displacement_untangled_model(filters_dy=24, filters_edm=16)
#model_dy.load_weights("/data/Images/MOP/data_segDis_resampled/seg_edm16dy24_model2_cp2.h5")

#model_edm = km.get_unet_model(image_shape=(256,32), filters=64, n_down=4)
#model_edm.load_weights("/data/Images/MOP/data_segDis/seg_edm64_model_cp.h5")

model = km.get_unet_model(image_shape=(256,32), filters=64, n_contractions=4, anisotropic_conv=True, n_input_channels=1, n_outputs=1, n_output_channels=1, out_activations=["sigmoid"], n_1x1_conv_after_decoder=0, use_1x1_conv_after_concat=False)
model.load_weights("/data/Images/DL_Models/delta_seg_cpV.h5")
shutil.rmtree("/data/Images/DL_Models/delta_seg")
export_model_bundle(model, "/data/Images/DL_Models/delta_seg")

# model = km.get_unet_model(image_shape=(256,32), filters=64, n_contractions=4, anisotropic_conv=True, n_inputs=2, n_input_channels=2, n_outputs=1, n_output_channels=3, out_activations=["softmax"], n_1x1_conv_after_decoder=0, use_1x1_conv_after_concat=False)
# model.load_weights("/data/Images/DL_Models/delta_track_cpV.h5")
# try:
#     shutil.rmtree("/data/Images/DL_Models/delta_track")
# except:
#     pass
# export_model_bundle(model, "/data/Images/DL_Models/delta_track")
