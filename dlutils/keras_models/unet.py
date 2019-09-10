#import numpy as np
from keras.models import Model
from keras.layers import Dense, Conv2D, Input, MaxPool2D, UpSampling2D, Concatenate, Conv2DTranspose, Dropout, Lambda
#import tensorflow as tf
from keras.optimizers import Adam
from keras import backend as K
from keras.losses import mean_squared_error
import os
from keras.preprocessing.image import array_to_img, img_to_array, load_img

def unet_down(input_layer, filters, kernel_size=3, pool=True, dropout=0, name=None):
    conv1 = Conv2D(filters, (kernel_size, kernel_size), padding='same', activation='relu', name=name+"_conv" if name else None)(input_layer)
    residual = Conv2D(filters, (kernel_size, kernel_size), padding='same', activation='relu', name=name+"_res" if name else None)(conv1)
    pool_input = Dropout(dropout, name=name+"_drop" if name else None)(residual) if dropout>0 else residual
    if pool:
        max_pool = MaxPool2D(pool_size=(2,2), name=name)(pool_input)
        return max_pool, residual
    else:
        return residual

def get_slice_channel_layer(channel):
    return Lambda(lambda x: x[:,:,:,channel:(channel+1)])

def unet_up(input_layer, residual, filters):
    filters=int(filters)
    upsample = UpSampling2D(size=(2,2))(input_layer)
    upconv = Conv2D(filters, kernel_size=(2, 2), padding="same")(upsample)
    concat = Concatenate(axis=3)([residual, upconv])
    conv1 = Conv2D(filters, (3, 3), padding='same', activation='relu')(concat)
    conv2 = Conv2D(filters, (3, 3), padding='same', activation='relu')(conv1)
    return conv2

class UnetEncoder():
    def __init__(self, n_down, n_filters, input, name="encoder"):
        #if isinstance(input, keras.layers.Input):
        #    self.layers=[input]
        #else:
        if min(input[0], input[1])/2**n_down<1:
            raise ValueError("too many down convolutions. minimal dimension is {}, number of convlution is {} shape in minimal dimension would be {}".format(min(input[0], input[1]), n_down, min(input[0], input[1])/2**n_down))
        self.layers=[Input(shape = input)]
        self.input_shape=input
        self.name=name
        self.residuals=[]
        self.n_filters=n_filters
        for i in range(n_down):
            self.down()
        self.make_feature_layer()

    def down(self):
        d, res = unet_down(self.layers[-1], self.get_n_filters(), kernel_size=self.get_kernel_size(), name=self.name+"_down"+str(len(self.layers)))
        self.layers.append(d)
        self.residuals.append(res)

    def make_feature_layer(self):
        features = unet_down(self.layers[-1], self.get_n_filters(), pool=False, kernel_size=self.get_kernel_size())
        self.layers.append(features)

    def get_kernel_size(self):
        min_dim = min(self.input_shape[0], self.input_shape[1])
        current_size = (int)(min_dim / 2**len(self.residuals))
        return 3 if current_size>=3 else current_size

    def get_n_filters(self):
        return self.n_filters * 2**len(self.residuals)

class UnetDecoder():
    def __init__(self, unet_encoder, input_layer=None):
        self.layers=[input_layer if input_layer is not None else unet_encoder.layers[-1]]
        self.unet_encoder = unet_encoder
        for i in range(len(unet_encoder.residuals)):
            self.up()

    def up(self):
        up = unet_up(self.layers[-1], residual=self.unet_encoder.residuals[-len(self.layers)], filters=self.get_n_filters())
        self.layers.append(up)

    def get_n_filters(self):
        n= self.unet_encoder.n_filters * 2**(len(self.unet_encoder.residuals) - len(self.layers))
        print("upconv: layer: {}/{}, # filters: {}".format(len(self.layers), len(self.unet_encoder.residuals), n))
        return n

def get_unet_model(input_shape, n_down, filters=64, n_output_channels=1, out_activations="linear"):
    encoder = UnetEncoder(n_down, filters, input_shape)
    decoder = UnetDecoder(encoder)
    if len(out_activations)==1 and n_output_channels>1:
         out_activations = [out_activations]*n_output_channels
    out = [Conv2D(filters=1, kernel_size=(1, 1), activation=out_activations[i])(decoder.layers[-1]) for i in range(n_output_channels)]
    if (n_output_channels==1):
        return Model(encoder.layers[0], out[0])
    else:
        return Model(encoder.layers[0], Concatenate(axis=3)(out))

def get_edm_displacement_model(filters=64, image_shape = (256, 32), edm_prop=0.5):
    encoder = UnetEncoder(4, filters, image_shape+(2,))
    decoder = UnetDecoder(encoder)
    edm = Conv2D(filters=1, kernel_size=(1, 1), activation="linear")(decoder.layers[-1])
    dy_input = unet_up(decoder.layers[-2], residual=encoder.residuals[0], filters=filters)
    dy = Conv2D(filters=1, kernel_size=(1, 1), activation="linear")(dy_input)
    out = Concatenate(axis=3)([edm, dy])

    model = Model(encoder.layers[0], out)

    def make_loss(y_true, y_pred, epsilon = 0.1, edm_prop=edm_prop):
      input_size = K.shape(y_true)[1] * K.shape(y_true)[2]
      batch_size = K.shape(y_true)[0]
      yt_edm = K.reshape(y_true[:,:,:, 0], (batch_size, input_size))
      yp_edm = K.reshape(y_pred[:,:,:, 0], (batch_size, input_size))
      yt_dis = K.reshape(y_true[:,:,:, 1], (batch_size, input_size))
      yp_dis = K.reshape(y_pred[:,:,:, 1], (batch_size, input_size))

      edm_loss = mean_squared_error(yt_edm, yp_edm)
      #dis_loss = K.mean( K.mean( (yt_edm + epsilon ) * K.square(yt_dis - yp_dis) , axis=-1) ) #/ # idée: relacher les contraintes aux bords des cellules
      #dis_loss = K.mean( K.sum( (yt_edm + epsilon ) * K.square(yt_dis - yp_dis) , axis=-1) / (K.sum(yt_edm + epsilon, axis=-1)) )
      dis_loss = mean_squared_error(yt_dis, yp_dis)
      return edm_prop * edm_loss + (1 - edm_prop) * dis_loss

    model.compile(optimizer=Adam(1e-3), loss=make_loss)
    return model

def get_edm_displacement_untangled_model(filters=16, image_shape = (256, 32), edm_prop=0.5):
    # make a regular Unet for edm regression
    edm_encoder = UnetEncoder(4, filters, image_shape+(1,))
    edm_decoder = UnetDecoder(edm_encoder) #edm_encoder.layers[-1].get_shape()[1:]
    edm = Conv2D(filters=1, kernel_size=(1, 1), activation="linear")(edm_decoder.layers[-1])
    edm_model_simple=Model(edm_encoder.layers[0], edm)
    edm_model_simple.compile(optimizer=Adam(1e-3), loss='mean_squared_error')
    edm_model = Model(edm_encoder.layers[0], [edm, edm_encoder.layers[-1]])

    # make a unet model for displacement with concatenated feature layer

    dy_encoder = UnetEncoder(4, filters, image_shape+(2,))
    input_prev = get_slice_channel_layer(0)(dy_encoder.layers[0])
    input_cur = get_slice_channel_layer(1)(dy_encoder.layers[0])
    edm_prev, edm_features_prev = edm_model(input_prev)
    edm_next, edm_features_next = edm_model(input_cur)
    dy_in = Concatenate(axis=3)([dy_encoder.layers[-1], edm_features_prev, edm_features_next])
    dy_in = Conv2D(filters=dy_encoder.get_n_filters(), kernel_size=(1, 1), activation="relu")(dy_in)

    dy_decoder = UnetDecoder(dy_encoder, input_layer = dy_in)
    dy = Conv2D(filters=1, kernel_size=(1, 1), activation="linear")(dy_decoder.layers[-1])
    out = Concatenate(axis=3)([edm_prev, edm_next, dy])
    print("out shape:", out.get_shape())
    dy_model =  Model(dy_encoder.layers[0], out)
    dy_model.compile(optimizer=Adam(1e-3), loss='mean_squared_error')
    return edm_model_simple, dy_model
