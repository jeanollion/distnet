from keras.layers import Layer, Dense, Reshape, Embedding, Concatenate, Conv2D
from keras.models import Model
import tensorflow as tf

class SelfAttention(Model):
    def __init__(self, d_model, max_spatial_dim, name="self_attention"):
        '''
            d_model : number of output channels
            max_spatial_dim : max number of spatial dimensions of input tensor (x * y). if > 0 enables positional encoding
            adapted from: https://www.tensorflow.org/tutorials/text/transformer
        '''
        super(SelfAttention, self).__init__(name=name)
        self.d_model = d_model

        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)

        if max_spatial_dim>0:
            self.pos_embedding = Embedding(max_spatial_dim, d_model)

    def call(self, x):
        '''
            x : tensor with shape (batch_size, y, x, channels)
        '''
        shape = tf.shape(x)
        batch_size = shape[0]
        spatial_dims = shape[1:-1]
        spatial_dim = tf.reduce_prod(spatial_dims)
        depth_dim = shape[3]

        if self.pos_embedding is not None:
            x_index = tf.range(spatial_dim, dtype=tf.int32)
            pos_emb = self.pos_embedding(x_index) # (spa_dim, d_model)
            pos_emb = tf.reshape(pos_emb, (spatial_dims[0], spatial_dims[1], self.d_model)) #for broadcasting purpose
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
        output = tf.reshape(scaled_attention, (batch_size, spatial_dims[0], spatial_dims[1], self.d_model))
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]+(self.d_model,)

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
