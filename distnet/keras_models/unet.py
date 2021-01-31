import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Cropping2D, Dense, Conv2D, Input, MaxPool2D, UpSampling2D, Concatenate, Conv2DTranspose, Dropout, SpatialDropout2D, SpatialDropout3D, Lambda, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mean_squared_error
import os
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from .self_attention import SelfAttention
from .attention_self_attention import AttentionSelfAttention
from .attention import Attention
from .multihead_self_attention import MultiHeadSelfAttention
from distnet.utils.helpers import ensure_multiplicity, flatten_list, append_to_list
from tensorflow.keras.backend import stop_gradient

def get_slice_channel_layer(channel, name=None): # tensorflow function !!
    return Lambda(lambda x: x[:,:,:,channel:(channel+1)], name = (name if name else "get_channel")+"_"+str(channel))

class UnetEncoder():
    def __init__(self, n_down, n_filters, max_filters=0, image_shape=None, anisotropic_conv=True, n_conv_layer_levels=2, n_1x1_conv_layer_levels=0, halve_filters_last_conv=False, num_attention_heads=0, add_attention=False, positional_encoding=True, dropout_contraction_levels=[], dropout_levels=[0.2], spatial_dropout=True, batch_norm=False, activation="relu", activation_1x1_conv_layer_levels='relu', padding='same', name="encoder"):
        if image_shape is not None:
            min_dim = min(image_shape[0], image_shape[1]) if image_shape[0] is not None and image_shape[1] is not None else image_shape[0] if image_shape[0] is not None else image_shape[1] if image_shape[1] is not None else None
            if min_dim is not None:
                assert min_dim/2**n_down>=1, "too many down convolutions. minimal dimension is {}, number of convlution is {} shape in minimal dimension would be {}".format(min_dim, n_down, min_dim/2**n_down)
        self.image_shape = image_shape
        self.padding=padding
        self.name=name
        self.layers=[]
        assert n_down>=0, "number of contractions should be >0"
        self.n_down = n_down
        self.n_filters=n_filters
        self.anisotropic_conv=anisotropic_conv
        self.spatial_dropout=spatial_dropout
        self.n_conv_layer_levels = ensure_multiplicity(n_down+1, n_conv_layer_levels)
        self.n_1x1_conv_layer_levels = ensure_multiplicity(n_down+1, n_1x1_conv_layer_levels)
        activation_1x1_conv_layer_levels = ensure_multiplicity(n_down+1, activation_1x1_conv_layer_levels)
        self.activation_1x1_conv_layer_levels = []
        for lidx, n_conv in enumerate(self.n_1x1_conv_layer_levels):
            self.activation_1x1_conv_layer_levels.append(ensure_multiplicity(n_conv, activation_1x1_conv_layer_levels[lidx]))
        if num_attention_heads>0: # remove one convolution at last layer to replace it by the self-attention layer
            self.n_conv_layer_levels[-1] = max(0, self.n_conv_layer_levels[-1] - 1)
        assert all([l>=0 for l in self.n_conv_layer_levels]), "convolution number should be >=0"
        self.activation_layer_level = []
        activation_level = ensure_multiplicity(self.n_down+1, activation)
        for lidx, n_conv in enumerate(self.n_conv_layer_levels):
            self.activation_layer_level.append(ensure_multiplicity(n_conv, activation_level[lidx]))
        self.num_attention_heads=num_attention_heads
        self.add_attention=add_attention
        self.positional_encoding=positional_encoding
        if max_filters<=0:
            self.max_filters = n_filters * 2**n_down
        else:
            self.max_filters=max_filters
        self.halve_filters_last_conv = halve_filters_last_conv
        self.batch_norm = batch_norm
        self.dropout_contraction_levels = [l if l>=0 else n_down + l + 1 for l in dropout_contraction_levels]
        assert all([l>=0 and l<=n_down for l in self.dropout_contraction_levels]), "dropout_contraction_level should be <={}".format(n_down)
        assert len(dropout_contraction_levels)==0 or len(dropout_contraction_levels)==len(dropout_levels), "dropout_contraction_levels & dropout_levels must have same length"
        self.dropout_levels = dropout_levels
        for layer_idx in range(n_down + 1): # +1 -> last feature layer
            self._make_layer(layer_idx)

    def encode(self, input):
        residuals = []
        if isinstance(input, list):
            input = Concatenate(axis=-1)(flatten_list(input))
        last_input = input
        for layer_idx in range(self.n_down):
            last_input, res = self._encode_layer(last_input, layer_idx)
            residuals.append(res)
        if self.num_attention_heads>0:
            feature_layer, attention_weights = self._encode_layer(last_input, self.n_down)
            return feature_layer, residuals, attention_weights
        else:
            feature_layer = self._encode_layer(last_input, self.n_down)
            return feature_layer, residuals

    def _make_layer(self, layer_idx):
        filters = self._get_n_filters(layer_idx)
        kernel_sizeX, kernel_sizeY = self._get_kernel_size(layer_idx)
        convLs = [ conv_block(filters//2 if self.halve_filters_last_conv and layer_idx==self.n_down and c==self.n_conv_layer_levels[layer_idx]-1 else filters, (kernel_sizeX, kernel_sizeY), activation=self.activation_layer_level[layer_idx][c], batch_norm=self.batch_norm, padding=self.padding, additional_1x1_convs=self.n_1x1_conv_layer_levels[layer_idx], additional_1x1_conv_activations=self.activation_1x1_conv_layer_levels[layer_idx], name="{}_l{}_conv{}".format(self.name, layer_idx, c) if self.name else None) for c in range(self.n_conv_layer_levels[layer_idx])]
        if len(convLs)==0 and self.positional_encoding and filters!=self._get_n_filters(layer_idx-1):
            raise ValueError("No convolution at last layer result in incompatibility in number of filters for positional encoding. Last layer should has {} filters while previous layer has {}".format(filters, self._get_n_filters(layer_idx-1)))
        dropoutL = self._get_dropout_layer(layer_idx)
        if layer_idx==self.n_down: # last layer: no maxpool. attention optional
            layers = convLs
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
            elif dropoutL is not None:
                layers.append(dropoutL)
            self.layers.append(layers)
        else: # normal layer
            max_poolL = MaxPool2D(pool_size=(2,2), name="{}_l{}_maxpool".format(self.name,layer_idx) if self.name else None)
            if dropoutL is not None:
                self.layers.append(flatten_list([convLs, dropoutL, max_poolL]))
            else:
                self.layers.append(flatten_list([convLs, max_poolL]))

    def _get_dropout_layer(self, layer_idx):
        if layer_idx in self.dropout_contraction_levels:
            dropout_level = self.dropout_levels[self.dropout_contraction_levels.index(layer_idx)]
            if self.spatial_dropout:
                if self.image_shape is None or len(self.image_shape)<3:
                    return SpatialDropout2D(dropout_level, name = "{}_l{}_dropout".format(self.name,layer_idx) if self.name else None)
                else:
                    return SpatialDropout3D(dropout_level, name = "{}_l{}_dropout".format(self.name,layer_idx) if self.name else None)
            else:
                return Dropout(dropout_level, name = "{}_l{}_dropout".format(self.name,layer_idx) if self.name else None)
        else:
            return None

    def _encode_layer(self, input, layer_idx):
        has_dropout = layer_idx in self.dropout_contraction_levels
        n_conv = self.n_conv_layer_levels[layer_idx]
        layers = self.layers[layer_idx]
        conv = input
        for convLidx in range(n_conv):
            conv = layers[convLidx](conv)
        if layer_idx==self.n_down: # last layer : no contraction
            if self.num_attention_heads>0:
                if self.add_attention:
                    def last_layer(conv):
                        attention, weights = layers[n_conv](conv)
                        if has_dropout:
                            attention = layers[n_conv+1](attention)
                        output = conv + attention
                        return output, weights
                    return last_layer(conv)
                else:
                    attention, weights = layers[n_conv](conv)
                    idx = n_conv + 1
                    if has_dropout:
                        attention = layers[idx](attention)
                        idx+=1
                    output = layers[idx]([conv, attention])
                    output = layers[idx+1](output)
                    return output, weights
            elif self.num_attention_heads==0:
                if has_dropout:
                    conv = layers[n_conv](conv)
                return conv
        else:
            residual = conv
            idx = n_conv
            if has_dropout:
                residual = layers[idx](residual)
                idx+=1
            max_pool = layers[idx](residual)
            return max_pool, residual

    def _get_image_shape(self, layer_idx):
        if self.image_shape is None:
            return None, None
        current_sizeX = self.image_shape[0] // 2**layer_idx if self.image_shape[0] is not None else None
        current_sizeY = self.image_shape[1] // 2**layer_idx if self.image_shape[1] is not None else None
        return current_sizeX, current_sizeY

    def _get_kernel_size(self, layer_idx):
        if self.image_shape is None:
            return 3
        current_sizeX, current_sizeY = self._get_image_shape(layer_idx)
        kerX = 3 if current_sizeX is None or current_sizeX>=3 else current_sizeX
        kerY = 3 if current_sizeY is None or current_sizeY>=3 else current_sizeY
        if self.anisotropic_conv:
            return kerX, kerY
        else:
            return min(kerX, kerY), min(kerX, kerY)

    def _get_n_filters(self, layer_idx):
        return min(self.n_filters * 2**layer_idx, self.max_filters)

class UnetDecoder():
    def __init__(self, n_up, n_filters, max_filters=0, image_shape=None,
                 anisotropic_conv=False, n_conv_layer_levels=2, n_1x1_conv_layer_levels=0, halve_filters_last_conv=False,
                 use_1x1_conv_after_concat=True, n_last_1x1_conv=0, omit_skip_connection_levels=[], stop_grad_skip_connection_levels=[], batch_norm = False,
                 upsampling_conv_kernel=2, use_transpose_conv=False, upsampling_filter_factor_levels=1, upsampling_bilinear=False, merge_mode_add=False,
                 activation='relu', upsampling_activation='relu', activation_1x1_conv_layer_levels='relu', last_1x1_activations='relu',
                 padding='same', valid_padding_crop=None, name="decoder"):
        self.layers=[]
        self.name=name
        assert n_up>=0, "number of upsampling should be >0"
        self.n_up = n_up
        self.n_filters = n_filters
        self.padding=padding
        if self.padding=='valid':
            assert valid_padding_crop is not None and len(valid_padding_crop)==n_up
            self.valid_padding_crop=valid_padding_crop[::-1]
        self.up_sampling_activation=upsampling_activation
        self.use_1x1_conv_after_concat=use_1x1_conv_after_concat
        self.batch_norm = batch_norm
        if max_filters<=0:
            max_filters = n_filters * 2**n_up
        self.n_conv_layer_levels = ensure_multiplicity(n_up, n_conv_layer_levels)[::-1]
        self.activation_layer_level = []
        activation_level = ensure_multiplicity(self.n_up, activation)[::-1]
        for lidx, n_conv in enumerate(self.n_conv_layer_levels):
            self.activation_layer_level.append(ensure_multiplicity(n_conv, activation_level[lidx]))
        self.n_1x1_conv_layer_levels = ensure_multiplicity(n_up, n_1x1_conv_layer_levels)[::-1]
        activation_1x1_conv_layer_levels = ensure_multiplicity(n_up, activation_1x1_conv_layer_levels)[::-1]
        self.activation_1x1_conv_layer_levels = []
        for lidx, n_conv in enumerate(self.n_1x1_conv_layer_levels):
            self.activation_1x1_conv_layer_levels.append(ensure_multiplicity(n_conv, activation_1x1_conv_layer_levels[lidx]))
        self.upsampling_filter_factor_levels = ensure_multiplicity(n_up, upsampling_filter_factor_levels)[::-1]
        self.max_filters=max_filters
        self.image_shape=image_shape
        self.upsampling_conv_kernel=upsampling_conv_kernel
        self.use_transpose_conv=use_transpose_conv
        self.anisotropic_conv=anisotropic_conv
        self.halve_filters_last_conv=halve_filters_last_conv
        self.merge_mode_add=merge_mode_add
        self.upsampling_bilinear=upsampling_bilinear
        if omit_skip_connection_levels is None:
            self.omit_skip_connection_levels = []
        else:
            self.omit_skip_connection_levels = [l if l>=0 else n_up + l for l in omit_skip_connection_levels]
        if stop_grad_skip_connection_levels is None:
            self.stop_grad_skip_connection_levels = []
        else:
            self.stop_grad_skip_connection_levels = [l if l>=0 else n_up + l for l in stop_grad_skip_connection_levels]

        for layer_idx in range(n_up):
            self._make_layer(self._get_n_filters(layer_idx), layer_idx)
        if not isinstance(n_last_1x1_conv, (list, tuple)):
            n_last_1x1_conv=[n_last_1x1_conv]
        last_1x1_activations = ensure_multiplicity(len(n_last_1x1_conv), last_1x1_activations)
        for b_idx, n_1x1 in enumerate(n_last_1x1_conv):
            last_1x1_activations[b_idx] = ensure_multiplicity(n_1x1, last_1x1_activations[b_idx])
        self.last_convs = [ [Conv2D(n_filters, (1, 1), padding='same', activation=last_1x1_activations[oidx][i], kernel_initializer = 'he_normal', name = "{}_conv1x1_{}_{}".format(self.name,oidx,i) if self.name else None) for i in range(n_last_1x1_conv[oidx])] for oidx in range(len(n_last_1x1_conv)) ]

    def _upsampling_block(self, filters, layer_idx):
        filter_factor = self.upsampling_filter_factor_levels[layer_idx]
        if self.use_transpose_conv and self.upsampling_conv_kernel>0:
            if self.batch_norm:
                upsampleL = Conv2DTranspose(int(filters*filter_factor), kernel_size=(self.upsampling_conv_kernel, self.upsampling_conv_kernel), strides=(2,2), padding='same', name = "{}_l{}_up".format(self.name, layer_idx) if self.name else None)
                batch_normL = BatchNormalization()
                activationL = Activation(self.up_sampling_activation)
                def upsampl_fun(batch):
                    up = upsampleL(batch)
                    bn = batch_normL(up)
                    return activationL(bn)
                return upsampl_fun
            else:
                return Conv2DTranspose(int(filters*filter_factor), kernel_size=(self.upsampling_conv_kernel, self.upsampling_conv_kernel), strides=(2,2), padding='same', activation = self.up_sampling_activation, name = "{}_l{}_up".format(self.name, layer_idx) if self.name else None)
        else:
            upsampleL = UpSampling2D(size=(2,2), interpolation='bilinear' if self.upsampling_bilinear else 'nearest', name = "{}_l{}_up".format(self.name, layer_idx) if self.name else None)
            if self.upsampling_conv_kernel<=0: # no convolution
                return upsampleL
            upconvL = conv_block(int(filters*filter_factor), (self.upsampling_conv_kernel, self.upsampling_conv_kernel), batch_norm=self.batch_norm, name = "{}_l{}_upconv".format(self.name, layer_idx) if self.name else None)
            def upsampl_fun(batch):
                up = upsampleL(batch)
                return upconvL(up)
            return upsampl_fun

    def _make_layer(self, filters, layer_idx):
        filters=int(filters)
        upsampleL = self._upsampling_block(filters, layer_idx)
        concatL = self.concat_block(layer_idx)
        ker_sizeX, ker_sizeY = self._get_kernel_size(layer_idx)
        convLs = [conv_block(filters//2 if self.halve_filters_last_conv and layer_idx<self.n_up-1 and c==self.n_conv_layer_levels[layer_idx]-1 else filters, (1 if c==0 and self.use_1x1_conv_after_concat else ker_sizeX, 1 if c==0 and self.use_1x1_conv_after_concat else ker_sizeY), batch_norm=self.batch_norm, padding=self.padding, activation=self.activation_layer_level[layer_idx][c], additional_1x1_convs=self.n_1x1_conv_layer_levels[layer_idx], additional_1x1_conv_activations=self.activation_1x1_conv_layer_levels[layer_idx], name="{}_l{}_conv{}".format(self.name, layer_idx, c) if self.name else None) for c in range(self.n_conv_layer_levels[layer_idx])]
        self.layers.append([upsampleL, concatL, convLs])

    def concat_block(self, layer_idx):
        name = "{}_l{}_concat".format(self.name, layer_idx) if self.name else None
        if self.merge_mode_add:
            concatL = Lambda(lambda array : tf.reduce_sum(tf.stack(array, axis=0), axis=0), name = name)
        else:
            concatL = Concatenate(axis=-1, name = name)
        if self.padding=='valid':
            def crop_and_concat(inputs):
                for i in range(len(inputs)-1):
                    inputs[i] = Cropping2D(cropping = self.valid_padding_crop[layer_idx])(inputs[i])
                return concatL(inputs)
            return crop_and_concat
        else:
            return concatL

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
            outputs = []
            for oidx, convs in enumerate(self.last_convs):
                output = last_input
                for conv in convs:
                    output = conv(output)
                outputs.append(output)
            if len(outputs)==1:
                outputs=outputs[0]
            last_input = outputs
            if return_all:
                all_activations[-1] = last_input
        if return_all:
            return all_activations
        else:
            return last_input

    def encode_and_decode(self, input, encoder):
        if encoder.n_down!=self.n_up:
            raise ValueError("encoder has {} enconding blocks whereas decoder has {} decoding blocks".format(enconder.n_down, self.n_up))
        if encoder.num_attention_heads>0:
            encoded, residuals, attention_weights = encoder.encode(input)
            return self.decode(encoded, residuals), input, attention_weights
        else:
            encoded, residuals = encoder.encode(input)
            return self.decode(encoded, residuals), input

    def _decode_layer(self, input, residual, layer_idx):
        layers = self.layers[layer_idx]
        upsample = layers[0](input)
        stop_grad = self.n_up - 1 - layer_idx in self.stop_grad_skip_connection_levels
        if self.n_up - 1 - layer_idx not in self.omit_skip_connection_levels:
            if isinstance(residual, list):
                if stop_grad:
                    residual = [stop_gradient(r) for r in residual]
                concat = layers[1](flatten_list(residual+[upsample]))
            else:
                if stop_grad:
                    residual = stop_gradient(residual)
                concat = layers[1]([residual, upsample])
        else:
            concat = upsample
        conv = concat
        for convL in layers[2]:
            conv = convL(conv)
        return conv

    def _get_kernel_size(self, layer_idx):
        if self.image_shape is None:
            return 3, 3
        current_sizeX = self.image_shape[0] // 2**(self.n_up - layer_idx - 1)  if self.image_shape[0] is not None else None
        current_sizeY = self.image_shape[1] // 2**(self.n_up - layer_idx - 1)  if self.image_shape[1] is not None else None
        kX = 3 if current_sizeX is None or current_sizeX>=3 else current_sizeX
        kY = 3 if current_sizeY is None or current_sizeY>=3 else current_sizeY
        if self.anisotropic_conv:
            return kX, kY
        else:
            return min(kX, kY), min(kX, kY)

    def _get_n_filters(self, layer_idx):
        return min(self.n_filters * 2**(self.n_up - layer_idx - 1), self.max_filters)

def conv_block(filters, kernel, batch_norm = False, activation='relu', kernel_initializer = 'he_normal', padding='same', additional_1x1_convs=0, additional_1x1_conv_activations='relu', name=None):
    additional_1x1_conv_activations = ensure_multiplicity(additional_1x1_convs, additional_1x1_conv_activations) if additional_1x1_convs>0 else []
    if not batch_norm:
         spa_conv = Conv2D(filters, kernel, padding=padding, activation=activation, kernel_initializer = kernel_initializer, name = name)
         if len(additional_1x1_conv_activations)==0:
             return spa_conv
         def conv_fun(input):
             res = spa_conv(input)
             for i, a in enumerate(additional_1x1_conv_activations):
                 res = Conv2D(filters, 1, padding='same', activation=a, kernel_initializer = kernel_initializer, name = name+"_1x1_{}".format(i))(res)
             return res
         return conv_fun
    else:
        convL = Conv2D(filters, kernel, padding=padding, kernel_initializer = kernel_initializer, name = name)
        batch_normL = BatchNormalization()
        activationL = Activation(activation)
        def conv_fun(input):
            c = convL(input)
            n = batch_normL(c)
            res = activationL(n)
            for i, a in enumerate(additional_1x1_conv_activations):
                res = Conv2D(filters, 1, padding='same', activation=a, kernel_initializer = kernel_initializer, name = name+"1x1_{}".format(i))(res)
        return conv_fun

def concat_and_conv(inputs, n_filters, layer_name, activation='relu'):
    concat = Concatenate(axis=3, name = layer_name+"_concat")(inputs)
    return Conv2D(n_filters, (1, 1), padding='same', activation=activation, kernel_initializer = 'he_normal', name = layer_name+"_conv1x1")(concat)

def get_distnet_model(image_shape=(256, 32), n_contractions=4, filters=128, max_filters=1024, n_outputs=3, n_output_channels=[1, 4, 2], out_activations=["linear", "softmax", "linear"], n_inputs=1, n_input_channels=2, num_attention_heads=1, positional_encoding=True, output_attention_weights=False, n_1x1_conv_after_decoder=1, dropout_contraction_levels=[-1], dropout_levels=0.2, spatial_dropout=False):
    return get_custom_unet_model(image_shape, n_contractions, filters, max_filters=max_filters, n_outputs=n_outputs, n_output_channels=n_output_channels, out_activations=out_activations, anisotropic_conv=True, n_inputs=n_inputs, n_input_channels=n_input_channels, use_1x1_conv_after_concat=True, use_self_attention=True, add_attention=False, num_attention_heads=num_attention_heads, output_attention_weights=output_attention_weights, n_1x1_conv_after_decoder=n_1x1_conv_after_decoder, dropout_contraction_levels=dropout_contraction_levels, dropout_levels=dropout_levels, spatial_dropout=spatial_dropout)

def get_unet_model(image_shape, n_contractions, filters, n_outputs=1, n_output_channels=1, out_activations=["linear"], n_inputs=1, n_input_channels=1, dropout_contraction_levels=[], dropout_levels=0.2, batch_norm=False):
    return get_custom_unet_model(image_shape=image_shape, n_contractions=n_contractions, filters=filters, max_filters=0, n_outputs=n_outputs, n_output_channels=n_output_channels, out_activations=out_activations, anisotropic_conv=False, n_inputs=n_inputs, n_input_channels=n_input_channels, use_1x1_conv_after_concat=False, n_1x1_conv_after_decoder=0, dropout_contraction_levels=dropout_contraction_levels, dropout_levels=dropout_levels, batch_norm=batch_norm)

def get_custom_unet_model(
    image_shape,
    n_contractions,
    filters,
    max_filters=0,
    n_outputs=1,
    n_output_channels=1,
    out_activations=["linear"],
    n_inputs=1,
    n_input_channels=1,
    anisotropic_conv=True,
    upsampling_conv_kernel=2,
    halve_filters_last_conv=False,
    use_self_attention=False,
    add_attention=False,
    num_attention_heads=1,
    positional_encoding=True,
    output_attention_weights=False,
    n_conv_layer_levels_encoder=2,
    n_conv_layer_levels_decoder=2,
    n_1x1_conv_after_decoder=0,
    encoder_n_1x1_conv_after_3x3_convs=0,
    decoder_n_1x1_conv_after_3x3_convs=0,
    use_1x1_conv_after_concat=True,
    use_transpose_conv=False,
    upsampling_bilinear=False,
    merge_mode_add = False,
    upsampling_filter_factor_levels=1,
    batch_norm=False,
    omit_skip_connection_levels=[],
    stop_grad_skip_connection_levels=[],
    encoder_conv_activations='relu',
    decoder_conv_activations='relu',
    decoder_last_1x1_activations='relu',
    decoder_1x1_conv_activations='relu',
    encoder_1x1_conv_activations='relu',
    n_stack=1,
    stacked_intermediate_outputs=True,
    stacked_skip_conection=True,
    dropout_contraction_levels=[],
    dropout_levels=0.2,
    spatial_dropout=True,
    residual=None,
    concatenate_outputs = False,
    padding='same', model_name = "UNet"):
    n_output_channels = ensure_multiplicity(n_outputs, n_output_channels)
    out_activations = ensure_multiplicity(n_outputs, out_activations)
    n_input_channels = ensure_multiplicity(n_inputs, n_input_channels)
    filters = ensure_multiplicity(2, filters)
    max_filters =  ensure_multiplicity(2, max_filters)
    dropout_levels = ensure_multiplicity(len(dropout_contraction_levels), dropout_levels)
    if residual is not None:
        for oidx, iidx in residual.items():
            assert iidx<n_inputs and oidx<n_outputs and n_input_channels[iidx]==n_output_channels[oidx], "invalid residual configuration: output and input indexes must exist and number of channel must be equal between input and output"
    if n_inputs>1:
        input = [Input(shape = image_shape+(n_input_channels[i],), name="input"+str(i)) for i in range(n_inputs)]
    else:
        input = Input(shape = image_shape+(n_input_channels[0],), name="input")
    crop, output_crop = get_valid_padding_crop(n_contractions, n_conv_e=n_conv_layer_levels_encoder, n_conv_d = [n- (1 if use_1x1_conv_after_concat else 0) for n in ensure_multiplicity(n_contractions, n_conv_layer_levels_decoder)])
    encoders = [UnetEncoder(n_contractions, filters[0], max_filters[0], image_shape, anisotropic_conv, n_conv_layer_levels_encoder, n_1x1_conv_layer_levels=encoder_n_1x1_conv_after_3x3_convs, halve_filters_last_conv=halve_filters_last_conv, num_attention_heads=num_attention_heads if use_self_attention else 0, add_attention=add_attention, positional_encoding=positional_encoding, dropout_contraction_levels=dropout_contraction_levels, dropout_levels=dropout_levels, spatial_dropout=spatial_dropout, batch_norm=batch_norm, activation=encoder_conv_activations, activation_1x1_conv_layer_levels=encoder_1x1_conv_activations, padding=padding, name="encoder{}".format(i)) for i in range(n_stack)]
    decoders = [UnetDecoder(n_contractions, filters[1], max_filters[1], image_shape, anisotropic_conv, n_conv_layer_levels_decoder, n_1x1_conv_layer_levels=decoder_n_1x1_conv_after_3x3_convs, halve_filters_last_conv=halve_filters_last_conv, n_last_1x1_conv=n_1x1_conv_after_decoder, use_1x1_conv_after_concat=use_1x1_conv_after_concat, omit_skip_connection_levels=omit_skip_connection_levels, stop_grad_skip_connection_levels=stop_grad_skip_connection_levels, upsampling_conv_kernel=upsampling_conv_kernel, use_transpose_conv=use_transpose_conv, upsampling_bilinear=upsampling_bilinear, merge_mode_add=merge_mode_add, upsampling_filter_factor_levels=upsampling_filter_factor_levels, batch_norm=batch_norm, activation=decoder_conv_activations, activation_1x1_conv_layer_levels=decoder_1x1_conv_activations, last_1x1_activations=decoder_last_1x1_activations, padding=padding, valid_padding_crop = crop, name="decoder{}".format(i)) for i in range(n_stack)]
    def get_output(layer, rank=0):
        name = "" if rank==0 else "_i"+str(rank)+"_"
        if n_outputs>1:
            if isinstance(layer, list):
                assert len(layer) == n_outputs, "number of output branches must coïncide with output number"
                outputs = [Conv2D(filters=n_output_channels[i], kernel_size=(1, 1), activation=out_activations[i], name="output"+name+str(i))(layer[i]) for i in range(n_outputs)]
            else:
                outputs = [Conv2D(filters=n_output_channels[i], kernel_size=(1, 1), activation=out_activations[i], name="output"+name+str(i))(layer) for i in range(n_outputs)]
            if residual is not None:
                for oidx in range(n_outputs):
                    if oidx in residual:
                        input_idx = residual[oidx]
                        if n_inputs==1:
                            if input_idx==0:
                                outputs[oidx]  =  outputs[oidx] + input
                        else:
                          outputs[oidx]  =  outputs[oidx] + input[input_idx]
            return outputs
        else:
            out =  Conv2D(filters=n_output_channels[0], kernel_size=(1, 1), activation=out_activations[0], name="output"+name)(layer)
            if residual is not None and 0 in residual:
                input_idx = residual[0]
                if n_inputs==1:
                    if input_idx==0:
                        return out + input
                else:
                    return out + input[input_idx]
            return out

    def get_intermediate_input(decoded_1, decoded_2, intermediate_outputs, rank):
        if (not stacked_skip_conection or decoded_1 is None) and not stacked_intermediate_outputs:
            return decoded_2
        concat = [decoded_2]
        if stacked_skip_conection and decoded_1 is not None:
            concat.append(decoded_1)
        if intermediate_outputs is not None:
            append_to_list(concat, intermediate_outputs)
        return concat_and_conv(concat, filters[1], "intermediate_"+str(rank), activation=activation)

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
    if concatenate_outputs:
        output = Concatenate(axis=3, name="output")(flatten_list(all_outputs))
        all_outputs = [output]
    if use_self_attention and output_attention_weights:
        all_outputs.append(attention_weights)
    model = Model(input, flatten_list(all_outputs), name=model_name)
    if padding == "valid":
        return model, output_crop
    else:
        return model

def get_valid_padding_crop(n_contractions, n_conv_e, n_conv_d, size=0, conv_radius=1):
    conv_radius = ensure_multiplicity(2, conv_radius)
    n_conv_e = ensure_multiplicity(n_contractions +1, n_conv_e)
    n_conv_d = ensure_multiplicity(n_contractions, n_conv_d)

    size_residual = []
    for i in range(n_contractions +1):
        start_size = size if i==0 else size_residual[-1] // 2
        size_residual.append(start_size - 2 * n_conv_e[i] * conv_radius[0])
    if n_contractions==0:
        return [], (size-size_residual[0])//2
    n_conv_d_rev = n_conv_d[::-1]
    size_concat = []
    for i in range(n_contractions):
        start_size = size_residual[-1] if i==0 else size_concat[-1] - 2 * n_conv_d_rev[i] * conv_radius[1]
        size_concat.append(start_size * 2)
    output_crop = ( size - ( size_concat[-1] - 2 * n_conv_d_rev[-1] * conv_radius[1] ) ) // 2
    size_concat = size_concat[::-1]
    crop = [ (size_residual[i] - size_concat[i] ) //2 for i in range(n_contractions)]
    return crop, output_crop
