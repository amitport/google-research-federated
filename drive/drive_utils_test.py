import tensorflow as tf
from absl.testing import parameterized

from drive.drive_utils import simulate_drive

SEED_PAIR = tf.constant([12345678, 87654321])


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
        rvec += simulate_drive(tf.identity(input),
                               sample_hadamard_seed=sample_hadamard_seed,
                               bits=bits,
                               hadamard_repeats=hadamard_repeats)

      rvec /= n_clients

      NMSE += tf.reduce_sum((input - rvec) ** 2) / tf.reduce_sum(input ** 2)
      print(f'NMSE={tf.reduce_sum((input - rvec) ** 2) / tf.reduce_sum(input ** 2)}')

    print(f'Average NMSE={NMSE / n_trials}')
    self.assertLessEqual(NMSE / n_trials, 0.001)
