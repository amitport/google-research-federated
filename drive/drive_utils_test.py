import tensorflow as tf
from absl.testing import parameterized

from drive.drive_utils import drive_roundtrip, drive_sub_roundtrip, \
  drive_sub_roundtrip2

SEED_PAIR = tf.constant([12345678, 87654321])
SEED_PAIR2 = tf.constant([87654321, 12345678])


class DriveUtilsTest(tf.test.TestCase, parameterized.TestCase):

  def test_dme(self):
    dim = 2 ** 19
    n_clients = 10
    bits = 4
    n_trials = 10
    hadamard_repeats = 1
    NMSE = 0

    for trial in range(n_trials):
      input = tf.random.normal((dim,))
      rvec = tf.zeros_like(input)

      for client in range(n_clients):
        sample_hadamard_seed = SEED_PAIR + (hadamard_repeats * client + hadamard_repeats * n_clients * trial)
        rvec += drive_roundtrip(tf.identity(input),
                               sample_hadamard_seed=sample_hadamard_seed,
                               bits=bits,
                               hadamard_repeats=hadamard_repeats)

      rvec /= n_clients

      NMSE += tf.reduce_sum((input - rvec) ** 2) / tf.reduce_sum(input ** 2)
      print(f'NMSE={tf.reduce_sum((input - rvec) ** 2) / tf.reduce_sum(input ** 2)}')

    print(f'Average NMSE={NMSE / n_trials}')
    self.assertLessEqual(NMSE / n_trials, 0.001)


  def test_sub_dme(self):
    dim = 2 ** 19
    n_clients = 1000
    bits = 1
    p = 0.9
    n_trials = 1
    hadamard_repeats = 1
    NMSE = 0

    for trial in range(n_trials):
      input = tf.random.normal((dim,))
      rvec = tf.zeros_like(input)

      for client in range(n_clients):
        sample_hadamard_seed = SEED_PAIR + (hadamard_repeats * client + hadamard_repeats * n_clients * trial)
        rand_p_seed = SEED_PAIR2 + (
            hadamard_repeats * client + hadamard_repeats * n_clients * trial)
        rvec += drive_sub_roundtrip2(tf.identity(input),
                               hadamard_seed=sample_hadamard_seed,
                               rand_p_seed=rand_p_seed,
                               p=p,
                               bits=bits,
                               hadamard_repeats=hadamard_repeats)

      rvec /= n_clients

      NMSE += tf.reduce_sum((input - rvec) ** 2) / tf.reduce_sum(input ** 2)
      print(f'NMSE={tf.reduce_sum((input - rvec) ** 2) / tf.reduce_sum(input ** 2)}')

    print(f'Average NMSE={NMSE / n_trials}')
    self.assertLessEqual(NMSE / n_trials, 0.001)