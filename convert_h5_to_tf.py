import sys
sys.path.append('/data/Dev/DL_Utils')
from dlutils.keras_models import get_edm_displacement_model, get_edm_displacement_untangled_model
from dlutils.utils import export_model_graph
model_edm, model_dy, model_dy_predict = get_edm_displacement_untangled_model(filters_dy=24, filters_edm=16)
model_dy.load_weights("/data/Images/MOP/data_segDis_resampled/seg_edm16dy24_model2_cp2.h5")
model_edm.load_weights("/data/Images/MOP/data_segDis_resampled/seg_edm16_model_lum_noise_cp2.h5")
export_model_graph(model_dy_predict, "/data/Images/MOP/data_segDis_resampled/", "model_edm_dy.pb")
