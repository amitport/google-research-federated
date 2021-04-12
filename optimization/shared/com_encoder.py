from tensorflow_model_optimization.python.core.internal import tensor_encoding as te
from tensorflow_model_optimization.python.core.internal.tensor_encoding.stages import HadamardEncodingStage, \
  FlattenEncodingStage, BitpackingEncodingStage
import tensorflow as tf

import math as m

HALF_NORMAL_MEAN_CONSTANT = m.sqrt(2 / m.pi)


# the expected center-of-mass in our case is: 1/sqrt(tensor_size) * HALF_NORMAL_MEAN_CONSTANT


@te.core.tf_style_encoding_stage
class HalfNormalCenterOfMassSign(te.core.EncodingStageInterface):
  ENCODED_VALUES_KEY = 'encoded'

  @property
  def name(self):
    """See base class."""
    return 'half_normal_com_sign'

  @property
  def compressible_tensors_keys(self):
    """See base class."""
    return [self.ENCODED_VALUES_KEY]

  @property
  def commutes_with_sum(self):
    """See base class."""
    return True

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
    return {self.ENCODED_VALUES_KEY: tf.cast(tf.greater(x, 0.), x.dtype)}

  def decode(self,
             encoded_tensors,
             decode_params,
             num_summands=None,
             shape=None):
    """See base class."""
    del decode_params, num_summands, shape  # Unused.

    X = encoded_tensors[self.ENCODED_VALUES_KEY] * 2. - 1.  # map {0, 1} to {-1, 1}
    # TODO center of mass `com` can possibly be created once in the constructor in the future (given known tensor size)
    # com = tf.math.rsqrt(tf.size(X, out_type=X.dtype)) * HALF_NORMAL_MEAN_CONSTANT
    # return X * com

    # no need to multiply by tf.math.rsqrt(tf.size(X, out_type=X.dtype))
    # since this happens in tf_utils.fast_walsh_hadamard_transform after the rotation
    return X * HALF_NORMAL_MEAN_CONSTANT


@te.core.tf_style_encoding_stage
class NormalizationStage(te.core.EncodingStageInterface):
  ENCODED_VALUES_KEY = 'encoded'
  NORM_KEY = 'norm'

  def __init__(self, ord="euclidean"):
    self.ord = ord

  @property
  def name(self):
    """See base class."""
    return 'normalization'

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
    del encode_params

    normalized, norm = tf.linalg.normalize(x, ord=self.ord)
    # print('origin: ', norm)

    return {
      # currently this is followed by sign so no point in normalizing
      self.ENCODED_VALUES_KEY: normalized,
      self.NORM_KEY: norm}

  def decode(self,
             encoded_tensors,
             decode_params,
             num_summands=None,
             shape=None):
    """See base class."""
    del decode_params, num_summands, shape  # Unused.
    # return tf.identity(encoded_tensors[self.NORMALIZED_VALUES_KEY])
    # normalized, _ = tf.linalg.normalize(encoded_tensors[self.NORMALIZED_VALUES_KEY], ord=self.ord)
    # print(_)
    # return normalized * encoded_tensors[self.NORM_KEY]
    normalized, _ = tf.linalg.normalize(encoded_tensors[self.ENCODED_VALUES_KEY], ord=self.ord)
    return normalized * encoded_tensors[self.NORM_KEY]
    # return (encoded_tensors[self.ENCODED_VALUES_KEY] / tf.norm(encoded_tensors[self.ENCODED_VALUES_KEY])) *
    # encoded_tensors[self.NORM_KEY]


def hadamard_com_encoder():
  encoder = te.core.EncoderComposer(BitpackingEncodingStage(1)).add_parent(
    HalfNormalCenterOfMassSign(), HalfNormalCenterOfMassSign.ENCODED_VALUES_KEY).add_parent(
    HadamardEncodingStage(), HadamardEncodingStage.ENCODED_VALUES_KEY).add_parent(    
    NormalizationStage(), NormalizationStage.ENCODED_VALUES_KEY).add_parent(
    FlattenEncodingStage(), FlattenEncodingStage.ENCODED_VALUES_KEY)

  return encoder.make()
