from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Conv2D, Input, Flatten, LeakyReLU
from tensorflow.keras.models import Model

def get_intensity_transformer(input_size=64, n_down=2, n_filters=4, n_filters_max=16, batch_mean=True, name="intensity_transformer", dtype="float32"):
    input = Input(shape = (input_size, input_size, 1), name="{}_input".format(name), dtype=dtype)
    conv = input
    filters = n_filters
    for l in range(n_down):
        conv = Conv2D(filters, kernel_size = (3,3), strides=2, padding='valid', activation=LeakyReLU(alpha=0.5), name="{}_conv_{}".format(name, l))(conv)
        filters *=2
        if n_filters_max>0:
            filters = min(filters, n_filters_max)
    conv_flat = Flatten()(conv)
    offset = Dense(1, name = "{}_offset".format(name))(conv_flat)
    scale = Dense(1, activation = "relu", name = "{}_scale".format(name))(conv_flat)
    if batch_mean:
        offset = K.mean(offset, axis=0)
        scale = K.mean(scale, axis=0)
    return Model(input, [offset, scale])

def plug_intensity_transformer(model, intensity_transformer_model, shared_input=True):
    if not shared_input:
        input = Input(shape = model.input.shape[1:], name="input_to_transform_"+model.name)
        thumb_input = intensity_transformer_model.input
        offset, scale = intensity_transformer_model.outputs
        scaled_input = ( input - offset ) * scale
        output = model(scaled_input)
        scaled_output = output / scale + offset
        return Model([input, thumb_input], scaled_output)
    else:
        input = Input(shape = intensity_transformer_model.input.shape[1:], name="input_to_transform_"+model.name)
        offset, scale = intensity_transformer_model(input)
        scaled_input = ( input - offset ) * scale
        output = model(scaled_input)
        scaled_output = output / scale + offset
        return Model(input, scaled_output)
