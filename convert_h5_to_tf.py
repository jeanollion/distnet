import sys
sys.path.append('/data/Dev/DL_Utils')
import dlutils.keras_models as km
from dlutils.utils import export_model_graph
#model_edm, model_dy, model_dy_predict = get_edm_displacement_untangled_model(filters_dy=24, filters_edm=16)
#model_dy.load_weights("/data/Images/MOP/data_segDis_resampled/seg_edm16dy24_model2_cp2.h5")
#model_edm.load_weights("/data/Images/MOP/data_segDis_resampled/seg_edm16_model_lum_noise_cp2.h5")

model_edm = km.get_unet_model(image_shape=(256,32), filters=64, n_down=4)
model_edm.load_weights("/data/Images/MOP/data_segDis/seg_edm64_model_cp.h5")
export_model_graph(model_edm, "/data/Images/MOP/data_segDis/", "model_edm64.pb")
