import tensorflow as tf
from tensorflow_model_optimization.python.core.internal import tensor_encoding as te


def _calculate_scale(x, x_estimation):
  # projected scale is ||x||^2 / <x, x_estimation>
  return tf.norm(x) ** 2 / tf.reduce_sum(x * x_estimation)


@te.core.tf_style_encoding_stage
class DriveSignStage(te.core.EncodingStageInterface):
  ENCODED_VALUES_KEY = 'encoded'
  SCALE_KEY = 'scale'

  @property
  def name(self):
    """See base class."""
    return 'drive_sign'

  @property
  def compressible_tensors_keys(self):
    """See base class."""
    return [self.ENCODED_VALUES_KEY]

  @property
  def commutes_with_sum(self):
    """See base class."""
    return False

  @property
  def decode_needs_input_shape(self):
    """See base class."""
    return False

  def get_params(self):
    """See base class."""
    return {}, {}

  def encode(self, x, encode_params):
    """See base class."""
    del encode_params  # Unused.

    scale = _calculate_scale(x, tf.sign(x))

    return {
      self.ENCODED_VALUES_KEY: tf.cast(x > 0.0, x.dtype),
      self.SCALE_KEY: scale,
    }

  def decode(self,
             encoded_tensors,
             decode_params,
             num_summands=None,
             shape=None):
    """See base class."""
    del decode_params, num_summands, shape  # Unused.

    onebit_signs = encoded_tensors[self.ENCODED_VALUES_KEY]
    scale = encoded_tensors[self.SCALE_KEY]

    signs = onebit_signs * 2.0 - 1.0

    return scale * signs