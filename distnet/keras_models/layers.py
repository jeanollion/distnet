from tensorflow import pad
from tensorflow.keras.layers import Layer, GlobalAveragePooling2D, Reshape, Conv2D, Multiply
from ..utils.denoise_utils import get_nd_gaussian_kernel

class ReflectionPadding2D(Layer):
  def __init__(self, padding=(1, 1), **kwargs):
    if not isinstance(padding, (list, tuple)):
      padding=(padding, padding)
    self.padding = tuple(padding)
    super().__init__(**kwargs)

  def compute_output_shape(self, input_shape):
    return (input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3])

  def call(self, input_tensor, mask=None):
    padding_width, padding_height = self.padding
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
