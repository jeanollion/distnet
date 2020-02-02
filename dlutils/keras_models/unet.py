import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Input, MaxPool2D, UpSampling2D, Concatenate, Conv2DTranspose, Dropout, Lambda, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mean_squared_error
import os
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from .self_attention import SelfAttention
from .attention_self_attention import AttentionSelfAttention
from .attention import Attention
from .multihead_self_attention import MultiHeadSelfAttention

def get_slice_channel_layer(channel, name=None): # tensorflow function !!
    return Lambda(lambda x: x[:,:,:,channel:(channel+1)], name = (name if name else "get_channel")+"_"+str(channel))

class UnetEncoder():
    def __init__(self, n_down, n_filters, max_filters=0, image_shape=None, anisotropic_conv=False, num_attention_heads=0, add_attention=False, positional_encoding=True, dropout_contraction_levels=[], dropout_levels=[0.2], name="encoder"):
        if image_shape!=None and min(image_shape[0], image_shape[1])/2**n_down<1:
            raise ValueError("too many down convolutions. minimal dimension is {}, number of convlution is {} shape in minimal dimension would be {}".format(min(image_shape[0], image_shape[1]), n_down, min(image_shape[0], image_shape[1])/2**n_down))
        self.image_shape = image_shape
        self.name=name
        self.layers=[]
        assert n_down>0, "number of contractions should be >0"
        self.n_down = n_down
        self.n_filters=n_filters
        self.anisotropic_conv=anisotropic_conv
        self.num_attention_heads=num_attention_heads
        self.add_attention=add_attention
        self.positional_encoding=positional_encoding
        if max_filters<=0:
            max_filters = n_filters * 2**n_down
        self.max_filters=max_filters
        self.dropout_contraction_levels = [l if l>=0 else n_down + l + 1 for l in dropout_contraction_levels]
        assert all([l>=0 and l<=n_down for l in self.dropout_contraction_levels]), "dropout_contraction_level should be <={}".format(n_down)
        assert len(dropout_contraction_levels)==0 or len(dropout_contraction_levels)==len(dropout_levels), "dropout_contraction_levels & dropout_levels must have same length"
        self.dropout_levels = dropout_levels
        for layer_idx in range(n_down + 1): # +1 -> last feature layer
            self._make_layer(layer_idx)

    def encode(self, input, layers_to_concatenate=None):
        if layers_to_concatenate and len(layers_to_concatenate)!=self.n_down+1:
            raise ValueError("{} layers to concatenate are provieded whereas {} are needed".format(len(layers_to_concatenate), self.n_down+1))
        residuals = []
        if isinstance(input, list):
            input = Concatenate(axis=3)(flatten_list(input))
        last_input = input
        for layer_idx in range(self.n_down):
            last_input, res = self._encode_layer(last_input, layer_idx, layers_to_concatenate[layer_idx] if layers_to_concatenate else None)
            residuals.append(res)
        if self.num_attention_heads>0:
            feature_layer, attention_weights = self._encode_layer(last_input, self.n_down, layers_to_concatenate[self.n_down] if layers_to_concatenate else None)
            return feature_layer, residuals, attention_weights
        else:
            feature_layer = self._encode_layer(last_input, self.n_down,  layers_to_concatenate[self.n_down] if layers_to_concatenate else None)
            return feature_layer, residuals

    def _make_layer(self, layer_idx):
        filters = self._get_n_filters(layer_idx)
        kernel_sizeX, kernel_sizeY = self._get_kernel_size(layer_idx)
        conv1L = Conv2D(filters, (kernel_sizeX, kernel_sizeY), padding='same', activation='relu', kernel_initializer = 'he_normal', name="{}_l{}_conv".format(self.name, layer_idx) if self.name else None)
        concatL=Concatenate(axis=3, name="{}_l{}_concat".format(self.name, layer_idx) if self.name else None)
        residualL = Conv2D(filters, (kernel_sizeX, kernel_sizeY), padding='same', activation='relu', kernel_initializer = 'he_normal', name="{}_l{}_res".format(self.name, layer_idx) if self.name else None)
        dropoutL = self._get_dropout_layer(layer_idx)
        if layer_idx==self.n_down: # last layer
            layers = [conv1L, concatL]
            if self.num_attention_heads>0:
                sx, sy = self._get_image_shape(layer_idx)
                if self.num_attention_heads==1:
                    selfAttentionL = SelfAttention(filters, [sx,sy], positional_encoding=self.positional_encoding, name="{}_l{}_self_attention".format(self.name, layer_idx) if self.name else None)
                else:
                    selfAttentionL = MultiHeadSelfAttention(filters, self.num_attention_heads, [sx,sy], positional_encoding=self.positional_encoding, name="{}_l{}_mh_self_attention".format(self.name,layer_idx) if self.name else None)
                layers.append(selfAttentionL)
                if dropoutL is not None:
                    layers.append(dropoutL)
                if not self.add_attention:
                    sa_concatL=Concatenate(axis=3, name="{}_{}_self_attention_concat".format(self.name, layer_idx) if self.name else None)
                    sa_conv1x1L = Conv2D(filters, (1, 1), padding='same', activation='relu', kernel_initializer = 'he_normal', name="{}_l{}_self_attention_conv1x1".format(self.name,layer_idx) if self.name else None)
                    layers.append(sa_concatL)
                    layers.append(sa_conv1x1L)
            else:
                layers.append(residualL)
                if self.num_attention_heads==0 and dropoutL is not None:
                    layers.append(dropoutL)
            self.layers.append(layers)
        else: # normal layer
            max_poolL = MaxPool2D(pool_size=(2,2), name="{}_l{}_maxpool".format(self.name,layer_idx) if self.name else None)
            if dropoutL is not None:
                self.layers.append([conv1L, concatL, residualL, dropoutL, max_poolL])
            else:
                self.layers.append([conv1L, concatL, residualL, max_poolL])

    def _get_dropout_layer(self, layer_idx):
        if layer_idx in self.dropout_contraction_levels:
            dropout_level = self.dropout_levels[self.dropout_contraction_levels.index(layer_idx)]
            return Dropout(dropout_level, name = "{}_l{}_dropout".format(self.name,layer_idx) if self.name else None)
        else:
            return None

    def _encode_layer(self, input, layer_idx, layers_to_concatenate=None):
        has_dropout = layer_idx in self.dropout_contraction_levels
        layers = self.layers[layer_idx]
        if layer_idx<self.n_down or self.num_attention_heads>=-1:
            conv1 = layers[0](input)
        else:
            conv1=input
        if layers_to_concatenate:
            conv1 = layers[1]([conv1]+layers_to_concatenate)
        if layer_idx==self.n_down: # last layer : no contraction
            if self.num_attention_heads>0:
                if self.add_attention:
                    def last_layer(conv1):
                        attention, weights = layers[2](conv1)
                        if has_dropout:
                            attention = layers[3](attention)
                        output = conv1 + attention
                        return output, weights
                    return last_layer(conv1)
                else:
                    attention, weights = layers[2](conv1)
                    idx = 3
                    if has_dropout:
                        attention = layers[idx](attention)
                        idx+=1
                    output = layers[idx]([conv1, attention])
                    idx+=1
                    output = layers[idx](output)
                    return output, weights
            elif self.num_attention_heads==0: # special case with only one covolution
                conv2 = layers[2](conv1)
                if has_dropout:
                    conv2 = layers[3](conv2)
                return conv2
            else: # no convolution at all
                return conv1 # input or concatenated input
        else:
            residual = layers[2](conv1)
            idx = 3
            if has_dropout:
                residual = layers[idx](residual)
                idx+=1
            max_pool = layers[idx](residual)
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
    def __init__(self, n_up, n_filters, max_filters=0, image_shape=None, anisotropic_conv=False, use_1x1_conv_after_concat=True, n_last_1x1_conv=0, name="decoder"):
        self.layers=[]
        self.name=name
        assert n_up>0, "number of upsampling should be >0"
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
        self.last_convs = [Conv2D(n_filters, (1, 1), padding='same', activation='relu', kernel_initializer = 'he_normal', name = "{}_conv1x1_{}".format(self.name,i) if self.name else None) for i in range(n_last_1x1_conv)]

    def _make_layer(self, filters, layer_idx):
        filters=int(filters)
        upsampleL = UpSampling2D(size=(2,2), name = "{}_l{}_up".format(self.name, layer_idx) if self.name else None)
        upconvL = Conv2D(filters, kernel_initializer = 'he_normal', kernel_size=(2, 2), padding="same", name = "{}_l{}_conv1".format(self.name, layer_idx) if self.name else None)
        concatL = Concatenate(axis=3, name = "{}_l{}_concat".format(self.name, layer_idx) if self.name else None)
        ker_sizeX, ker_sizeY = self._get_kernel_size(layer_idx)
        kX = 1 if self.use_1x1_conv_after_concat else ker_sizeX
        kY = 1 if self.use_1x1_conv_after_concat else ker_sizeY
        conv1L = Conv2D(filters, (kX, kY), padding='same', activation='relu', kernel_initializer = 'he_normal', name = "{}_l{}_conv2".format(self.name, layer_idx) if self.name else None)
        conv2L = Conv2D(filters, (ker_sizeX, ker_sizeY), padding='same', activation='relu', kernel_initializer = 'he_normal', name = "{}_l{}_conv3".format(self.name, layer_idx) if self.name else None)
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
                all_activations[-1] = last_input
        if return_all:
            return all_activations
        else:
            return last_input

    def encode_and_decode(self, input, encoder, layers_to_concatenate=None):
        if encoder.n_down!=self.n_up:
            raise ValueError("encoder has {} enconding blocks whereas decoder has {} decoding blocks".format(enconder.n_down, self.n_up))
        if encoder.num_attention_heads>0:
            encoded, residuals, attention_weights = encoder.encode(input, layers_to_concatenate)
            return self.decode(encoded, residuals), input, attention_weights
        else:
            encoded, residuals = encoder.encode(input, layers_to_concatenate)
            return self.decode(encoded, residuals), input

    def _decode_layer(self, input, residual, layer_idx):
        layers = self.layers[layer_idx]
        upsample = layers[0](input)
        upconv = layers[1](upsample)
        if isinstance(residual, list):
            concat = layers[2](flatten_list(residual+[upconv]))
        else:
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

def get_unet_model(image_shape, n_contractions, filters=64, max_filters=0, n_outputs=1, n_output_channels=1, out_activations=["linear"], anisotropic_conv=False, n_inputs=1, n_input_channels=1, use_self_attention=False, add_attention=False, num_attention_heads=1, positional_encoding=True, output_attention_weights=False, n_1x1_conv_after_decoder=0, use_1x1_conv_after_concat=True, n_stack=1, stacked_intermediate_outputs=True, stacked_skip_conection=True, dropout_contraction_levels=[], dropout_levels=0.2):
    n_output_channels = _ensure_multiplicity(n_outputs, n_output_channels)
    out_activations = _ensure_multiplicity(n_outputs, out_activations)
    n_input_channels = _ensure_multiplicity(n_inputs, n_input_channels)
    filters = _ensure_multiplicity(2, filters)
    max_filters =  _ensure_multiplicity(2, max_filters)
    dropout_levels = _ensure_multiplicity(len(dropout_contraction_levels), dropout_levels)
    encoders = [UnetEncoder(n_contractions, filters[0], max_filters[0], image_shape, anisotropic_conv, num_attention_heads if use_self_attention else 0, add_attention, positional_encoding=positional_encoding, dropout_contraction_levels=dropout_contraction_levels, dropout_levels=dropout_levels, name="encoder"+str(i)+"_") for i in range(n_stack)]
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
    if use_self_attention:
        dec, _input, attention_weights = decoders[0].encode_and_decode(input, encoders[0]) # if several inputs _input = concatenated tensor
    else:
        dec, _input = decoders[0].encode_and_decode(input, encoders[0]) # if several inputs _input = concatenated tensor
    decoded_layers = [dec]
    for i in range(1, n_stack):
        if stacked_intermediate_outputs:
            all_outputs.append(get_output(decoded_layers[-1], rank=i))
        intermediate_input = get_intermediate_input(decoded_layers[-2] if i>1 else None, decoded_layers[-1], all_outputs[-1] if stacked_intermediate_outputs else None, i)
        if use_self_attention:
            decoded, _, attention_weights = decoders[i].encode_and_decode(intermediate_input, encoders[i])
        else:
            decoded, _ = decoders[i].encode_and_decode(intermediate_input, encoders[i])
        decoded_layers.append(decoded)

    all_outputs.append(get_output(decoded_layers[-1]))
    if use_self_attention and output_attention_weights:
        all_outputs.append(attention_weights)
    return Model(input, flatten_list(all_outputs))

def get_unet_plus_plus_model(image_shape, n_contractions, filters=64, max_filters=0, anisotropic_conv=True, n_outputs=1, n_output_channels=1, out_activations=["linear"], n_inputs=1, n_input_channels=1, use_self_attention=False, num_attention_heads=1, dropout_contraction_levels=[], dropout_levels=0.2, decoder_contraction_level=[]):
    n_output_channels = _ensure_multiplicity(n_outputs, n_output_channels)
    out_activations = _ensure_multiplicity(n_outputs, out_activations)
    n_input_channels = _ensure_multiplicity(n_inputs, n_input_channels)
    filters = _ensure_multiplicity(2, filters)
    max_filters =  _ensure_multiplicity(2, max_filters)
    dropout_levels = _ensure_multiplicity(len(dropout_contraction_levels), dropout_levels)
    encoder = UnetEncoder(n_contractions, filters[0], max_filters[0], image_shape, anisotropic_conv, num_attention_heads if use_self_attention else 0, add_attention=False, positional_encoding=True, dropout_contraction_levels=dropout_contraction_levels, dropout_levels=dropout_levels, name="encoder")
    if decoder_contraction_level is None or len(decoder_contraction_level)==0:
        decoder_contraction_level = list(range(n_contractions))
    else:
        decoder_contraction_level = [l if l>=0 else n_contractions + l for l in decoder_contraction_level]
        decoder_contraction_level.sort()
        print(decoder_contraction_level)
        assert n_contractions-1 in decoder_contraction_level, "last contraction level must be decoded"
        assert all([l>=0 and l<=n_contractions for l in decoder_contraction_level]), "contraction level should be <={}".format(n_contractions)
    decoders = [UnetDecoder(c+1, filters[1], max_filters[1], image_shape, anisotropic_conv, n_last_1x1_conv=1, use_1x1_conv_after_concat=True, name="decoder{}".format(c)) for c in decoder_contraction_level]

    if n_inputs>1:
        input = [Input(shape = image_shape+(n_input_channels[i],), name="input"+str(i)) for i in range(n_inputs)]
    else:
        input = Input(shape = image_shape+(n_input_channels[0],), name="input")

    if use_self_attention and num_attention_heads>0:
        encoded, residuals, attention_weights = encoder.encode(input, None)
    else:
        encoded, residuals = encoder.encode(input, None)
    residuals.append(encoded)

    decoded = list()
    for i, (d_idx, decoder) in enumerate(zip(decoder_contraction_level, decoders)):
        all_residuals = list()
        for r_idx in range(0, d_idx+1):
            current_residuals = [residuals[r_idx]]
            for prev_i in range(i-1, -1, -1):
                dec = decoded[prev_i]
                if len(dec)>r_idx: # decoder yields enough residuals
                    current_residuals.append(dec[-(r_idx+1)])
            all_residuals.append(current_residuals)
            #getshape = lambda l : [getshape(r) if type(r)==list else r.shape for r in l]
            #print("decoder: {}, residual: {}: residual shapes: {}".format(d_idx, r_idx, getshape(current_residuals)))
        decoded.append(decoders[i].decode(residuals[d_idx+1], all_residuals, True))
        #print("decoder: {}, decoded shapes: {}".format(d_idx,  getshape(decoded[-1])))
    def get_output(layer, rank=0):
        if n_outputs>1:
            return [Conv2D(filters=n_output_channels[i], kernel_size=(1, 1), activation=out_activations[i], name="output{}_{}".format(rank,i))(layer) for i in range(n_outputs)]
        else:
            return Conv2D(filters=n_output_channels[0], kernel_size=(1, 1), activation=out_activations[0], name="output{}".format(rank))(layer)
    outputs = [get_output(d[-1], i) for i,d in enumerate(decoded)]
    return Model(input, outputs)

def get_attention_tracking_model(image_shape, n_contractions, filters=[32, 64], max_filters=1024, n_outputs=1, n_output_channels=1, out_activations=["linear"], anisotropic_conv=False, self_attention=False, add_attention=False, num_attention_heads=1, output_attention_weights=False, n_1x1_conv_after_decoder=0, use_1x1_conv_after_concat=True):
    n_output_channels = _ensure_multiplicity(n_outputs, n_output_channels)
    out_activations = _ensure_multiplicity(n_outputs, out_activations)
    filters = _ensure_multiplicity(2, filters)
    max_filters =  _ensure_multiplicity(2, max_filters)
    encoder = UnetEncoder(n_contractions, filters[0], max_filters[0], image_shape, anisotropic_conv, -2 if self_attention else -1, False, False, name="encoder")
    decoder = UnetDecoder(n_contractions, filters[1], max_filters[1], image_shape, anisotropic_conv, n_last_1x1_conv=n_1x1_conv_after_decoder, use_1x1_conv_after_concat=use_1x1_conv_after_concat, name="decoder")

    input = Input(shape = image_shape+(2,), name="input")
    [input_prev, input_cur] = tf.unstack(input, 2, axis=-1)
    input_prev = tf.expand_dims(input_prev, -1)
    input_cur = tf.expand_dims(input_cur, -1)

    def get_output(layer, rank=0):
        name = "" if rank==0 else "_i"+str(rank)+"_"
        if n_outputs>1:
            return [Conv2D(filters=n_output_channels[i], kernel_size=(1, 1), activation=out_activations[i], name="output"+name+str(i))(layer) for i in range(n_outputs)]
        else:
            return Conv2D(filters=n_output_channels[0], kernel_size=(1, 1), activation=out_activations[0], name="output"+name)(layer)

    enc_prev, res_prev = encoder.encode(input_prev) # prev
    enc_cur, res_cur = encoder.encode(input_cur) # cur
    #residuals = [[res_prev[i], res_cur[i]] for i in range(len(res_cur))]
    residuals = res_cur
    sx, sy = encoder._get_image_shape(n_contractions)
    n_filters_att = encoder._get_n_filters(n_contractions)
    if self_attention:
        attentionL = AttentionSelfAttention(n_filters_att, [sx,sy], name=encoder.name+str(n_contractions+1)+"_attention_selfattention")
    else:
        attentionL = Attention(n_filters_att, [sx,sy], name=encoder.name+str(n_contractions+1)+"_attention")
    attention, attention_weights = attentionL([enc_prev, enc_cur])
    if add_attention:
        encoded = attention + enc_cur
    else:
        a_concatL=Concatenate(axis=3, name=encoder.name+str(n_contractions+1)+"_attention_concat")
        a_conv1x1L = Conv2D(n_filters_att, (1, 1), padding='same', activation='relu', kernel_initializer = 'he_normal', name=encoder.name+str(n_contractions+1)+"_attention_conv1x1")
        concat = a_concatL([attention, enc_cur])
        encoded = a_conv1x1L(concat)
    decoded = decoder.decode(encoded, residuals)

    all_outputs = []
    all_outputs.append(get_output(decoded))
    if output_attention_weights:
        all_outputs.append(attention_weights)
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
     elif n==0:
         return []
     return object
