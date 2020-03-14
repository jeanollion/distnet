import sys
sys.path.append('/data/Dev/DL_Utils')
import distnet.keras_models as km
from distnet.utils.helpers import export_model_bundle
import shutil
import tensorflow as tf
print(tf.__version__)

model = km.get_distnet_model()
model.load_weights("/data/Images/DL_Models/distNet_weights.h5")
export_model_bundle(model, "/data/Images/DL_Models/model_bact_phase_save5", overwrite=True)
