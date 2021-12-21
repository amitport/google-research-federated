import tensorflow as tf
from absl.testing import parameterized

from distributed_dp import compression_utils
from distributed_dp.drive_utils import drive_quantization, inverse_drive_quantization

SEED_PAIR = (12345678, 87654321)


class DriveUtilsTest(tf.test.TestCase, parameterized.TestCase):

  def test_dme(self):
    dim = 2 ** 19
    n_clients = 10
    bits = 4
    n_trials = 10

    NMSE = 0
    for trial in range(n_trials):
      input = tf.random.normal((dim,))
      rvec = tf.zeros_like(input)

      for client in range(n_clients):
        rot_x = compression_utils.randomized_hadamard_transform(input, seed_pair=[SEED_PAIR[0] + trial,
                                                                                  SEED_PAIR[1] + n_clients],
                                                                repeat=1)
        x, scale = drive_quantization(rot_x, bits=bits)
        x = inverse_drive_quantization(x, scale, bits=bits)
        x = compression_utils.inverse_randomized_hadamard_transform(x, original_dim=dim,
                                                                    seed_pair=[SEED_PAIR[0] + trial,
                                                                               SEED_PAIR[1] + n_clients])

        rvec += x

      rvec /= n_clients

      NMSE += tf.reduce_sum((input - rvec) ** 2) / tf.reduce_sum(input ** 2)
      print(f'NMSE={NMSE}')

    self.assertAllClose(NMSE / n_trials, 0.)
