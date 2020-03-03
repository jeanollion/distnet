import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Reshape, Embedding, Concatenate, Conv2D
from tensorflow.keras.models import Model
import numpy as np

class SelfAttention(Model):
    def __init__(self, d_model, spatial_dims, positional_encoding=True, name="self_attention"):
        '''
            d_model : number of output channels
            spatial_dim : spatial dimensions of input tensor (x , y)
            if positional_encoding: depth must correspond to input channel number
            adapted from: https://www.tensorflow.org/tutorials/text/transformer
        '''
        super().__init__(name=name)
        self.d_model = d_model
        self.spatial_dims=spatial_dims
        self.spatial_dim = np.prod(spatial_dims)
        self.wq = Dense(self.d_model, name=name+"_q")
        self.wk = Dense(self.d_model, name=name+"_k")
        self.wv = Dense(self.d_model, name=name+"_w")
        self.positional_encoding=positional_encoding
        if positional_encoding:
            self.pos_embedding = Embedding(self.spatial_dim, d_model, name=name+"pos_enc") # TODO test other positional encoding. in particular that encodes X and Y

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
            pos_emb = tf.reshape(pos_emb, (self.spatial_dims[0], self.spatial_dims[1], self.d_model)) #for broadcasting purpose
            x = x + pos_emb # broadcast

        q = self.wq(x)  # (batch_size, *spa_dims, d_model)
        k = self.wk(x)  # (batch_size, *spa_dims, d_model)
        v = self.wv(x)  # (batch_size, *spa_dims, d_model)

        q = tf.reshape(q, (batch_size, -1, depth_dim)) # (batch_size, spa_dim, d_model)
        k = tf.reshape(k, (batch_size, -1, depth_dim))
        v = tf.reshape(v, (batch_size, -1, depth_dim))

        # scaled_attention.shape == (batch_size, spa_dims, depth)
        # attention_weights.shape == (batch_size, spa_dims, spa_dims)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v)
        output = tf.reshape(scaled_attention, (batch_size, self.spatial_dims[0], self.spatial_dims[1], self.d_model))
        tf.identity(attention_weights, name=self.name+"_attention_weights")
        return output, attention_weights

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]+(self.d_model,), (input_shape[0],self.spatial_dim,self.spatial_dim)

def scaled_dot_product_attention(q, k, v):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)

    Returns:
    output, attention_weights

    from : https://www.tensorflow.org/tutorials/text/transformer
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights
