from tensorflow import pad
from tensorflow.keras.layers import Layer, GlobalAveragePooling2D, Reshape, Conv2D, Multiply, Conv3D
from distnet.utils.denoise_utils import get_nd_gaussian_kernel
from ..utils.helpers import ensure_multiplicity
from tensorflow.python.keras.engine.input_spec import InputSpec
import tensorflow as tf

class ReflectionPadding2D(Layer):
  def __init__(self, paddingYX=(1, 1), **kwargs):
    if not isinstance(paddingYX, (list, tuple)):
      padding=(paddingYX, paddingYX)
    self.padding = tuple(padding)
    super().__init__(**kwargs)

  def compute_output_shape(self, input_shape):
    return (input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3])

  def call(self, input_tensor, mask=None):
    padding_height, padding_width = self.padding
    return pad(input_tensor, [[0,0], [padding_height, padding_height], [padding_width, padding_width], [0,0] ], 'REFLECT')

  def get_config(self):
      config = super().get_config().copy()
      config.update({"padding": self.padding})
      return config

class Gaussian2D(Layer):
  def __init__(self, radius=1, **kwargs):
    self.radius = radius
    super().__init__(**kwargs)

  def build(self, input_shape):
    n_chan = input_shape[-1]
    gauss_ker = get_nd_gaussian_kernel(radius=self.radius, ndim=2)[...,np.newaxis, np.newaxis]
    kernel = tf.constant(gauss_ker, dtype=tf.float32)
    if n_chan>1:
      self.kernel = tf.tile(kernel, [1, 1, n_chan, 1])
    else:
      self.kernel = kernel
    self.pointwise_filter = tf.eye(n_chan, batch_shape=[1, 1])

  def compute_output_shape(self, input_shape):
    return (input_shape[0], input_shape[1] - self.radius * 2, input_shape[2] - self.radius * 2, input_shape[3])

  def call(self, input_tensor, mask=None):
    return tf.nn.separable_conv2d(input_tensor, self.kernel, self.pointwise_filter, strides=[1, 1, 1, 1], padding='VALID')

  def get_config(self):
    config = super().get_config().copy()
    config.update({"radius": self.radius})
    return config

def channel_attention(n_filters, activation='relu'): # TODO TEST + make layer or model + set name to layers
  def ca_fun(input):
    gap = GlobalAveragePooling2D()(input)
    gap = Reshape((1, 1, n_filters))(gap) # or use dense layers and reshape afterwards
    conv1 = Conv2D(kernel_size=1, filters = n_filters, activation=activation)(gap)
    key = Conv2D(kernel_size=1, filters = n_filters, activation='sigmoid')(conv1)
    return Multiply()([key, input])
  return ca_fun

class Conv3D_YXC(Layer):
  def __init__(self, filters, kernelYX, n_channels, padding="REFLECT", **kwargs):
    kernelYXC=ensure_multiplicity(2, kernelYX)
    if padding=="same":
        padding = "CONSTANT"
    name = kwargs.pop('name', None)
    self.padding_constant_value = kwargs.pop('constant_values', 0)
    self.convL = Conv3D(filters=filters, kernel_size = (kernelYX[0], kernelYX[1], n_channels), padding="valid",  name = name+"conv" if name is not None else None, **kwargs)
    self.input_spec = InputSpec(ndim=4)
    self.padding = padding
    super().__init__(name)

  def compute_output_shape(self, input_shape):
    if self.padding=="valid":
        return (input_shape[0], input_shape[1] - self.convL.kernel_size[0] + 1 , input_shape[2] - self.convL.kernel_size[1] + 1, self.filters)
    else:
        return (input_shape[0], input_shape[1], input_shape[2], self.filters)

  def call(self, input_tensor, mask=None):
    if self.padding!="valid":
        padding_height, padding_width = [ (k-1)//2 for k in self.convL.kernel_size[:-1]]
        input_tensor = pad(input_tensor, [[0,0], [padding_height, padding_height], [padding_width, padding_width], [0,0] ], mode = self.padding, constant_values=self.padding_constant_value, name = self.name+"pad" if self.name is not None else None)
    conv = self.convL(input_tensor[...,tf.newaxis]) # add "channel" axis for 5D tensor
    return conv[:, :, :, 0, :] # valid padding on last conv axis -> size 1

# TODO get_config -> attributes of convL ?
