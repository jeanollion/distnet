import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Input, MaxPool2D, UpSampling2D, Concatenate, Conv2DTranspose, Dropout, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mean_squared_error
import os
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from .self_attention import SelfAttention
from .multihead_self_attention import MultiHeadSelfAttention

def get_slice_channel_layer(channel, name=None): # tensorflow function !!
    return Lambda(lambda x: x[:,:,:,channel:(channel+1)], name = (name if name else "get_channel")+"_"+str(channel))

class UnetEncoder():
    def __init__(self, n_down, n_filters, max_filters=0, image_shape=None, anisotropic_conv=False, num_attention_heads=0, name="encoder_"):
        if image_shape!=None and min(image_shape[0], image_shape[1])/2**n_down<1:
            raise ValueError("too many down convolutions. minimal dimension is {}, number of convlution is {} shape in minimal dimension would be {}".format(min(image_shape[0], image_shape[1]), n_down, min(image_shape[0], image_shape[1])/2**n_down))
        self.image_shape = image_shape
        self.name=name
        self.layers=[]
        self.n_down = n_down
        self.n_filters=n_filters
        self.anisotropic_conv=anisotropic_conv
        self.num_attention_heads=num_attention_heads
        if max_filters<=0:
            max_filters = n_filters * 2**n_down
        self.max_filters=max_filters
        for layer_idx in range(n_down + 1): # +1 -> last feature layer
            self._make_layer(layer_idx)

    def encode(self, input, layers_to_concatenate=None):
        if layers_to_concatenate and len(layers_to_concatenate)!=self.n_down+1:
            raise ValueError("{} layers to concatenate are provieded whereas {} are needed".format(len(layers_to_concatenate), self.n_down+1))
        residuals = []
        last_input = input
        for layer_idx in range(self.n_down):
            last_input, res = self._encode_layer(last_input, layer_idx, layers_to_concatenate[layer_idx] if layers_to_concatenate else None)
            residuals.append(res)
        feature_layer = self._encode_layer(last_input, self.n_down,  layers_to_concatenate[self.n_down] if layers_to_concatenate else None)
        return feature_layer, residuals

    def _make_layer(self, layer_idx):
        filters = self._get_n_filters(layer_idx)
        kernel_sizeX, kernel_sizeY = self._get_kernel_size(layer_idx)
        conv1L = Conv2D(filters, (kernel_sizeX, kernel_sizeY), padding='same', activation='relu', kernel_initializer = 'he_normal', name=self.name+str(layer_idx+1)+"_conv" if self.name else None)
        concatL=Concatenate(axis=3, name=self.name+str(layer_idx+1)+"_concat" if self.name else None)
        residualL = Conv2D(filters, (kernel_sizeX, kernel_sizeY), padding='same', activation='relu', kernel_initializer = 'he_normal', name=self.name+str(layer_idx+1)+"_res" if self.name else None)
        if len(self.layers)==self.n_down: # last layer
            if self.num_attention_heads>0:
                sx, sy = self._get_image_shape(layer_idx)
                if self.num_attention_heads==1:
                    selfAttentionL = SelfAttention(filters, [sx,sy], name=self.name+str(layer_idx+1)+"_self_attention" if self.name else None)
                else:
                    selfAttentionL = MultiHeadSelfAttention(filters, self.num_attention_heads, [sx,sy], name=self.name+str(layer_idx+1)+"_mh_self_attention" if self.name else None)
                # TODO concat or average ?
                sa_concatL=Concatenate(axis=3, name=self.name+str(layer_idx+1)+"self_attention_concat" if self.name else None)
                sa_conv1x1L = Conv2D(filters, (1, 1), padding='same', activation='relu', kernel_initializer = 'he_normal', name=self.name+str(layer_idx+1)+"_self_attention_conv1x1" if self.name else None)
                self.layers.append([conv1L, concatL, selfAttentionL, sa_concatL, sa_conv1x1L])
            else:
                self.layers.append([conv1L, concatL, residualL])
        else:
            max_poolL = MaxPool2D(pool_size=(2,2), name=self.name+str(layer_idx+1) if self.name else None)
            self.layers.append([conv1L, concatL, residualL, max_poolL])

    def _encode_layer(self, input, layer_idx, layers_to_concatenate=None):
        layers = self.layers[layer_idx]
        conv1 = layers[0](input)
        if layers_to_concatenate:
            conv1 = layers[1]([conv1]+layers_to_concatenate)
        residual = layers[2](conv1)
        if layer_idx==self.n_down: # last layer : no contraction
            if self.num_attention_heads>0:
                output = layers[3]([conv1, residual])
                return layers[4](output)
            else:
                return residual
        else:
            max_pool = layers[3](residual)
            return max_pool, residual

    def _get_image_shape(self, layer_idx):
        if self.image_shape is None:
            return None, None
        current_sizeX = self.image_shape[0] // 2**layer_idx
        current_sizeY = self.image_shape[1] // 2**layer_idx
        return current_sizeX, current_sizeY

    def _get_kernel_size(self, layer_idx):
        if self.image_shape is None:
            return 3
        current_sizeX, current_sizeY = self._get_image_shape(layer_idx)
        kerX = 3 if current_sizeX>=3 else current_sizeX
        kerY = 3 if current_sizeY>=3 else current_sizeY
        if self.anisotropic_conv:
            return kerX, kerY
        else:
            return min(kerX, kerY), min(kerX, kerY)

    def _get_n_filters(self, layer_idx):
        return min(self.n_filters * 2**layer_idx, self.max_filters)

class UnetDecoder():
    def __init__(self, n_up, n_filters, max_filters=0, image_shape=None, anisotropic_conv=False, use_1x1_conv_after_concat=True, n_last_1x1_conv=0, name="decoder_"):
        self.layers=[]
        self.name=name
        self.n_up = n_up
        self.n_filters = n_filters
        self.use_1x1_conv_after_concat=use_1x1_conv_after_concat
        if max_filters<=0:
            max_filters = n_filters * 2**n_up
        self.max_filters=max_filters
        self.image_shape=image_shape
        self.anisotropic_conv=anisotropic_conv
        for layer_idx in range(n_up):
            self._make_layer(self._get_n_filters(layer_idx), layer_idx)
        self.last_convs = [Conv2D(n_filters, (1, 1), padding='same', activation='relu', kernel_initializer = 'he_normal', name = self.name+"conv1x1_"+str(i+1) if self.name else None) for i in range(n_last_1x1_conv)]

    def _make_layer(self, filters, layer_idx):
        filters=int(filters)
        upsampleL = UpSampling2D(size=(2,2), name = self.name+str(layer_idx+1)+"_up" if self.name else None)
        upconvL = Conv2D(filters, kernel_initializer = 'he_normal', kernel_size=(2, 2), padding="same", name = self.name+str(layer_idx+1)+"_conv1" if self.name else None)
        concatL = Concatenate(axis=3, name = self.name+str(layer_idx+1)+"_concat" if self.name else None)
        ker_sizeX, ker_sizeY = self._get_kernel_size(layer_idx)
        kX = 1 if self.use_1x1_conv_after_concat else ker_sizeX
        kY = 1 if self.use_1x1_conv_after_concat else ker_sizeY
        conv1L = Conv2D(filters, (kX, kY), padding='same', activation='relu', kernel_initializer = 'he_normal', name = self.name+str(layer_idx+1)+"_conv2" if self.name else None)
        conv2L = Conv2D(filters, (ker_sizeX, ker_sizeY), padding='same', activation='relu', kernel_initializer = 'he_normal', name = self.name+str(layer_idx+1)+"_conv3" if self.name else None)
        self.layers.append([upsampleL, upconvL, concatL, conv1L, conv2L])

    def decode(self, input, residuals, return_all=False):
        if len(residuals)!=self.n_up:
            raise ValueError("#{} residuals are provided whereas {} are needed".format(len(residuals), self.n_up))
        last_input = input
        all_activations = []
        for layer_idx in range(self.n_up):
            last_input = self._decode_layer(last_input, residuals[-layer_idx-1], layer_idx)
            if return_all:
                all_activations.append(last_input)
        if len(self.last_convs)>0:
            for conv in self.last_convs:
                last_input = conv(last_input)
                if return_all:
                    all_activations.append(last_input)

        if return_all:
            return all_activations
        else:
            return last_input

    def encode_and_decode(self, input, encoder, layers_to_concatenate=None):
        if encoder.n_down!=self.n_up:
            raise ValueError("encoder has {} enconding blocks whereas decoder has {} decoding blocks".format(enconder.n_down, self.n_up))
        if isinstance(input, list):
            input = Concatenate(axis=3)(input)
        encoded, residuals = encoder.encode(input, layers_to_concatenate)
        return (self.decode(encoded, residuals), input)

    def _decode_layer(self, input, residual, layer_idx):
        layers = self.layers[layer_idx]
        upsample = layers[0](input)
        upconv = layers[1](upsample)
        concat = layers[2]([residual, upconv])
        conv1 = layers[3](concat)
        conv2 = layers[4](conv1)
        return conv2

    def _get_kernel_size(self, layer_idx):
        if self.image_shape is None:
            return 3
        current_sizeX = self.image_shape[0] // 2**(self.n_up - layer_idx - 1)
        current_sizeY = self.image_shape[1] // 2**(self.n_up - layer_idx - 1)
        kX = 3 if current_sizeX>=3 else current_sizeX
        kY = 3 if current_sizeY>=3 else current_sizeY
        if self.anisotropic_conv:
            return kX, kY
        else:
            return min(kX, kY), min(kX, kY)

    def _get_n_filters(self, layer_idx):
        return min(self.n_filters * 2**(self.n_up - layer_idx - 1), self.max_filters)

def concat_and_conv(inputs, n_filters, layer_name):
    concat = Concatenate(axis=3, name = layer_name+"_concat")(inputs)
    return Conv2D(n_filters, (1, 1), padding='same', activation='relu', kernel_initializer = 'he_normal', name = layer_name+"_conv1x1")(concat)

def get_unet_model(image_shape, n_contractions, filters=64, max_filters=0, n_outputs=1, n_output_channels=1, out_activations=["linear"], anisotropic_conv=False, n_inputs=1, n_input_channels=1, use_self_attention=False, num_attention_heads=1, n_1x1_conv_after_decoder=0, use_1x1_conv_after_concat=True, n_stack=1, stacked_intermediate_outputs=True, stacked_skip_conection=True):
    n_output_channels = _ensure_multiplicity(n_outputs, n_output_channels)
    out_activations = _ensure_multiplicity(n_outputs, out_activations)
    n_input_channels = _ensure_multiplicity(n_inputs, n_input_channels)
    filters = _ensure_multiplicity(2, filters)
    max_filters =  _ensure_multiplicity(2, max_filters)
    encoders = [UnetEncoder(n_contractions, filters[0], max_filters[0], image_shape, anisotropic_conv, num_attention_heads if use_self_attention else 0, name="encoder"+str(i)+"_") for i in range(n_stack)]
    decoders = [UnetDecoder(n_contractions, filters[1], max_filters[1], image_shape, anisotropic_conv, n_last_1x1_conv=n_1x1_conv_after_decoder, use_1x1_conv_after_concat=use_1x1_conv_after_concat, name="decoder"+str(i)+"_") for i in range(n_stack)]

    if n_inputs>1:
        input = [Input(shape = image_shape+(n_input_channels[i],), name="input"+str(i)) for i in range(n_inputs)]
    else:
        input = Input(shape = image_shape+(n_input_channels[0],), name="input")

    def get_output(layer, rank=0):
        name = "" if rank==0 else "_i"+str(rank)+"_"
        if n_outputs>1:
            return [Conv2D(filters=n_output_channels[i], kernel_size=(1, 1), activation=out_activations[i], name="output"+name+str(i))(layer) for i in range(n_outputs)]
        else:
            return Conv2D(filters=n_output_channels[0], kernel_size=(1, 1), activation=out_activations[0], name="output"+name)(layer)

    def get_intermediate_input(decoded_1, decoded_2, intermediate_outputs, rank):
        if (not stacked_skip_conection or decoded_1 is None) and not stacked_intermediate_outputs:
            return decoded_2
        concat = [decoded_2]
        if stacked_skip_conection and decoded_1 is not None:
            concat.append(decoded_1)
        if intermediate_outputs is not None:
            append_to_list(concat, intermediate_outputs)
        return concat_and_conv(concat, filters[1], "intermediate_"+str(rank))

    all_outputs = []
    dec, _input = decoders[0].encode_and_decode(input, encoders[0]) # if several inputs _input = concatenated tensor
    decoded_layers = [dec]
    for i in range(1, n_stack):
        if stacked_intermediate_outputs:
            all_outputs.append(get_output(decoded_layers[-1], rank=i))
        intermediate_input = get_intermediate_input(decoded_layers[-2] if i>1 else None, decoded_layers[-1], all_outputs[-1] if stacked_intermediate_outputs else None, i)
        decoded, _ = decoders[i].encode_and_decode(intermediate_input, encoders[i])
        decoded_layers.append(decoded)

    all_outputs.append(get_output(decoded_layers[-1]))
    return Model(input, flatten_list(all_outputs))

def flatten_list(l):
    flat_list = []
    for item in l:
        append_to_list(flat_list, item)
    return flat_list

def append_to_list(l, element):
    if isinstance(element, list):
        l.extend(element)
    else:
        l.append(element)

def _ensure_multiplicity(n, object):
     if not isinstance(object, list):
         object = [object]
     if len(object)>1 and len(object)!=n:
         raise ValueError("length should be either 1 either equal to n"+str(n))
     if n>1 and len(object)==1:
         object = object*n
     return object
