import functools
from absl import logging
import tensorflow_federated as tff

from drive.drive_utils import _create_drive_fn, _create_drive_sub_fn
from optimization2.compressors_lib.hadamard import _create_hadamard_fn
from optimization2.compressors_lib.model_opt_wrapper import _create_kashin_fn
from optimization2.compressors_lib.sq import _create_sq_fn

SUPPORTED_COMPRESSORS = ['noop', 'drive', 'drive_sub', 'hadamard', 'kashin', 'sq']


def get_compressor_factory(compressor: str, **kwargs):
  logging.info(f'compressor {compressor}!')
  logging.info(kwargs)
  if compressor == 'noop':
    def _create_noop_fn(value_type):
      @tff.tf_computation(value_type)
      def noop_fn(record):
        return record

      return noop_fn

    return _create_noop_fn
  elif compressor == 'drive':
    bits = kwargs['num_bits']
    create_compress_roundtrip_fn = functools.partial(_create_drive_fn,
                                                     bits=bits,
                                                     hadamard_repeats=1)
    return create_compress_roundtrip_fn
  elif compressor == 'drive_sub':
    bits = kwargs['num_bits']
    p = kwargs['p']
    create_compress_roundtrip_fn = functools.partial(_create_drive_sub_fn,
                                                     bits=bits,
                                                     p=p,
                                                     hadamard_repeats=1)
    return create_compress_roundtrip_fn
  elif compressor == 'hadamard':
    bits = kwargs['num_bits']
    p = kwargs['p']
    create_compress_roundtrip_fn = functools.partial(_create_hadamard_fn,
                                                     bits=bits,
                                                     p=p,
                                                     hadamard_repeats=1)
    return create_compress_roundtrip_fn

  elif compressor == 'kashin':
    bits = kwargs['num_bits']
    p = kwargs['p']
    create_compress_roundtrip_fn = functools.partial(_create_kashin_fn,
                                                     bits=bits,
                                                     p=p)
    return create_compress_roundtrip_fn
  elif compressor == 'sq':
    bits = kwargs['num_bits']
    p = kwargs['p']
    create_compress_roundtrip_fn = functools.partial(_create_sq_fn,
                                                     bits=bits,
                                                     p=p)
    return create_compress_roundtrip_fn
  else:
    raise ValueError('expected a compressor name')
