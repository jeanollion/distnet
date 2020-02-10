import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Reshape, Embedding, Concatenate, Conv2D
from tensorflow.keras.models import Model
import numpy as np
from .self_attention import scaled_dot_product_attention

class MultiHeadSelfAttention(Model):
    def __init__(self, depth, num_heads, spatial_dims, positional_encoding=True, name="mh_self_attention"):
        '''
            depth : number of output channels
            spatial_dim : spatial dimensions of input tensor (x , y)
            if positional_encoding: depth must correspond to input channel number
            adapted from: https://www.tensorflow.org/tutorials/text/transformer
        '''
        super().__init__(name=name)
        assert num_heads>0, "invalid attention head number"
        self.num_heads = num_heads
        self.depth = depth
        self.d_model = depth * self.num_heads
        self.spatial_dims=spatial_dims
        self.spatial_dim = np.prod(spatial_dims)
        self.wq = Dense(self.d_model, name=name+"_q")
        self.wk = Dense(self.d_model, name=name+"_k")
        self.wv = Dense(self.d_model, name=name+"_w")
        self.dense = Dense(depth, name=name+"_lin")
        self.positional_encoding=positional_encoding
        if positional_encoding:
            self.pos_embedding = Embedding(self.spatial_dim, self.depth, name=name+"pos_enc")

    def split_heads(self, x, batch_size):
      """Split the last dimension into (num_heads, depth).
      Transpose the result such that the shape is (batch_size, num_heads, spa_dim, depth)
      """
      x = tf.reshape(x, (batch_size, self.spatial_dim, self.num_heads, self.depth))
      return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x):
        '''
            x : tensor with shape (batch_size, y, x, channels)
        '''
        shape = tf.shape(x)
        batch_size = shape[0]
        #spatial_dims = shape[1:-1]
        #spatial_dim = tf.reduce_prod(spatial_dims)
        depth_dim = shape[3]

        if self.positional_encoding:
            x_index = tf.range(self.spatial_dim, dtype=tf.int32)
            pos_emb = self.pos_embedding(x_index) # (spa_dim, d_model)
            pos_emb = tf.reshape(pos_emb, (self.spatial_dims[0], self.spatial_dims[1], self.depth)) #for broadcasting purpose
            x = x + pos_emb # broadcast (depth mut be equals to depth_dim)

        q = self.wq(x)  # (batch_size, *spa_dims, depth*num_heads)
        k = self.wk(x)  # (batch_size, *spa_dims, depth*num_heads)
        v = self.wv(x)  # (batch_size, *spa_dims, depth*num_heads)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, spa_dim, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, spa_dim, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, spa_dim, depth)

        # scaled_attention.shape == (batch_size, num_heads, spa_dim, depth)
        # attention_weights.shape == (batch_size, num_heads, spa_dim, spa_dim)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v)
        tf.identity(attention_weights, name=self.name+"_attention_weights")
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, spa_dims, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, spa_dims, depth*num_heads)
        output = self.dense(concat_attention)  # (batch_size, spa_dim, depth)
        output = tf.reshape(output, (batch_size, self.spatial_dims[0], self.spatial_dims[1], self.depth))
        return output, attention_weights

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]+(self.depth,), (input_shape[0], self.num_heads, self.spatial_dim, self.spatial_dim)
