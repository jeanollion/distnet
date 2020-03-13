import sys
sys.path.append('/data/Dev/DL_Utils')
import distnet.keras_models as km
from distnet.utils.helpers import export_model_bundle
import shutil
import tensorflow as tf
print(tf.__version__)

model = km.get_unet_model(image_shape=(256,32), filters=64, n_contractions=4, anisotropic_conv=True, n_input_channels=1, n_outputs=1, n_output_channels=1, out_activations=["sigmoid"], n_1x1_conv_after_decoder=0, use_1x1_conv_after_concat=False)
model = km.get_distnet_model()
model.load_weights("/data/Images/DL_Models/distNet_weights.h5")
export_model_bundle(model, "/data/Images/DL_Models/model_bact_phase_save5", overwrite=True)
