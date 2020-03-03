import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Reshape, Embedding, Concatenate, Conv2D
from tensorflow.keras.models import Model
import numpy as np
from .self_attention import scaled_dot_product_attention

class AttentionSelfAttention(Model):
    def __init__(self, depth, spatial_dims, num_heads=1, positional_encoding=True, name="attention_self_attention"):
        '''
            depth : number of output channels
            spatial_dim : spatial dimensions of input tensor (x , y)
            if positional_encoding: depth must correspond to input channel number
            adapted from: https://www.tensorflow.org/tutorials/text/transformer
        '''
        super().__init__(name=name)
        self.num_heads = num_heads
        self.depth = depth
        self.spatial_dims=spatial_dims
        self.spatial_dim = np.prod(spatial_dims)
        self.wq = Dense(self.depth * num_heads *2, name=name+"_q")
        self.wka = Dense(self.depth * num_heads, name=name+"_ka")
        self.wva = Dense(self.depth * num_heads, name=name+"_va")
        self.wksa = Dense(self.depth * num_heads, name=name+"_ksa")
        self.wvsa = Dense(self.depth * num_heads, name=name+"_vsa")
        self.dense = Dense(self.depth, name=name+"_lin")
        self.positional_encoding=positional_encoding
        if positional_encoding:
            self.pos_embedding = Embedding(self.spatial_dim, self.depth, name=name+"pos_enc")

    def split_heads(self, x, batch_size, num_heads):
      """Split the last dimension into (num_heads, depth).
      Transpose the result such that the shape is (batch_size, num_heads, spa_dim, depth)
      """
      x = tf.reshape(x, (batch_size, self.spatial_dim, num_heads, self.depth))
      return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x):
        '''
            x : list of 2 tensor with shape (batch_size, y, x, channels)
        '''
        [kvx, qx] = x
        shape = tf.shape(qx)
        batch_size = shape[0]
        #spatial_dims = shape[1:-1]
        #spatial_dim = tf.reduce_prod(spatial_dims)
        depth_dim = shape[3]

        if self.positional_encoding:
            x_index = tf.range(self.spatial_dim, dtype=tf.int32)
            pos_emb = self.pos_embedding(x_index) # (spa_dim, d_model)
            pos_emb = tf.reshape(pos_emb, (self.spatial_dims[0], self.spatial_dims[1], self.depth)) #for broadcasting purpose
            qx = qx + pos_emb # broadcast (depth mut be equals to depth_dim)
            kvx = kvx + pos_emb # broadcast (depth mut be equals to depth_dim)

        q = self.wq(qx)  # (batch_size, *spa_dims, depth*num_heads*2)
        ka = self.wka(kvx)  # (batch_size, *spa_dims, depth*num_heads)
        va = self.wva(kvx)  # (batch_size, *spa_dims, depth*num_heads)
        ksa = self.wksa(qx)  # (batch_size, *spa_dims, depth*num_heads)
        vsa = self.wvsa(qx)  # (batch_size, *spa_dims, depth*num_heads)

        q = self.split_heads(q, batch_size, self.num_heads*2) # (batch_size, num_heads*2, spa_dim, depth)
        ka = self.split_heads(ka, batch_size, self.num_heads)  # (batch_size, num_heads, spa_dim, depth)
        va = self.split_heads(va, batch_size, self.num_heads)  # (batch_size, num_heads, spa_dim, depth)
        ksa = self.split_heads(ksa, batch_size, self.num_heads)  # (batch_size, num_heads, spa_dim, depth)
        vsa = self.split_heads(vsa, batch_size, self.num_heads)  # (batch_size, num_heads, spa_dim, depth)
        k = tf.concat([ka, ksa], 1)  # (batch_size, num_heads * 2, spa_dim, depth)
        v = tf.concat([va, vsa], 1)  # (batch_size, num_heads * 2, spa_dim, depth)

        # scaled_attention.shape == (batch_size, num_heads, spa_dim, depth)
        # attention_weights.shape == (batch_size, num_heads, spa_dim, spa_dim)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v)
        tf.identity(attention_weights, name=self.name+"_attention_selfattention_weights")
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, spa_dims, num_heads*2, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.depth * self.num_heads*2))  # (batch_size, spa_dims, depth*num_heads*2)
        output = self.dense(concat_attention)  # (batch_size, spa_dim, depth)
        output = tf.reshape(output, (batch_size, self.spatial_dims[0], self.spatial_dims[1], self.depth))
        return output, attention_weights

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]+(self.depth,), (input_shape[0], self.num_heads * 2, self.spatial_dim, self.spatial_dim)
