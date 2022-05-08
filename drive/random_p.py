import tensorflow as tf

from tensorflow_model_optimization.python.core.internal import \
    tensor_encoding as te
from tensorflow_model_optimization.python.core.internal.tensor_encoding.core import \
    EncoderComposer


def sample_indices(shape, p, seed):
    mask = tf.less(tf.random.stateless_uniform(shape, seed=seed), p)
    return tf.where(mask)


@te.core.tf_style_encoding_stage
class RandomPEncodingStage(te.core.EncodingStageInterface):
    ENCODED_VALUES_KEY = 'non_zero_floats'
    SEED_VALUES_KEY = 'seed'

    def __init__(self, p):
        self._p = p

    def encode(self, x, encode_params):
        seed = tf.random.uniform([2], minval=tf.int64.min, maxval=tf.int64.max,
                                 dtype=tf.int64)

        flat_x = tf.reshape(x, [-1])
        indices = sample_indices(flat_x.shape, self._p, seed)
        vals = tf.gather(flat_x, indices) * (1 / self._p)

        return {self.ENCODED_VALUES_KEY: vals,
                self.SEED_VALUES_KEY: seed}

    def decode(self,
               encoded_tensors,
               decode_params,
               num_summands=None,
               shape=None):
        del num_summands  # Unused.

        vals = encoded_tensors[self.ENCODED_VALUES_KEY]
        seed = encoded_tensors[self.SEED_VALUES_KEY]
        flat_x_shape = tf.expand_dims(tf.reduce_prod(shape), 0)

        indices = sample_indices(flat_x_shape, self._p, seed)

        decoded_values = tf.scatter_nd(
            tf.expand_dims(indices, 1),
            vals,
            tf.cast(flat_x_shape, indices.dtype)
        )
        return tf.reshape(decoded_values, shape)

    def get_params(self):
        return {}, {}

    @property
    def name(self):
        return 'random_p_encoding_stage'

    @property
    def compressible_tensors_keys(self):
        return [self.ENCODED_VALUES_KEY]

    @property
    def commutes_with_sum(self):
        return False

    @property
    def decode_needs_input_shape(self):
        return True


class Enc_Wrapper:
  def __init__(self, enc):
    self.enc = enc

  def encode_decode(self, x):
    self.state = self.enc.initial_state()
    encode_params, decode_params = self.enc.get_params(self.state)
    encoded_tensors, state_update_tensors, input_shapes = self.enc.encode(x, encode_params)

    decoded_x = self.enc.decode(encoded_tensors, decode_params, input_shapes)
    self.state = self.enc.update_state(self.state, state_update_tensors)
    return decoded_x


if __name__ == '__main__':
    enc = Enc_Wrapper(EncoderComposer(RandomPEncodingStage(p=0.1)).make())

    enc.encode_decode(tf.reshape(tf.cast(tf.range(100), tf.float32), [2, 10, 5]))
