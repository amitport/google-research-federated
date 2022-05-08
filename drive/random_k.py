import time

import tensorflow as tf
from tqdm import trange
import numpy as np


def numpy_choice(n, k, seed):
  # x will be a numpy array with the contents of the input to the
  rng = np.random.default_rng()#seed)
  return rng.choice(n, size=k, replace=False)


def random_choice_without_replacement(n, k, seed):
    return tf.py_function(func=numpy_choice,
                   inp=[n, k, 42],
                   Tout=tf.int64)

# def random_choice_without_replacement(n, k, seed):
#     """equivalent to 'numpy.random.choice(n, size=k, replace=False)'"""
#     return tf.math.top_k(tf.random.stateless_uniform(shape=[n], seed=seed), k,
#                          sorted=False).indices


# def random_choice_without_replacement2(n, k, seed):
#     """equivalent to 'numpy.random.choice(n, size=k, replace=False)'"""
#     return tf.random.shuffle(tf.range(n))[:k]


def select_random_k(x, k, seed):
    d = tf.size(x)

    # the scale for getting an unbiased estimate is 1/p
    scale = tf.cast(d / k, x.dtype)
    return scale * tf.gather(x, random_choice_without_replacement(d, k, seed))


def unselect_random_k(x, original_size, seed):
    indices = random_choice_without_replacement(original_size, tf.size(x), seed=seed)
    return tf.scatter_nd(tf.expand_dims(indices, -1), x, shape=[original_size])


def random_k_roundtrip(x, k):
    seed = tf.random.uniform([2], maxval=tf.int64.max, dtype=tf.int64)
    n = tf.size(x)

    encoded = select_random_k(x, k, seed)
    return unselect_random_k(encoded, n, seed)


def test_dme(n_trials, n_clients, dim, k):
    start = time.time()
    NMSE = 0
    for _ in range(n_trials):
        vec = tf.random.normal([dim])

        v_hat_sum = tf.zeros_like(vec)
        for _ in trange(n_clients):
            v_hat_sum += random_k_roundtrip(vec, k)
        v_hat_mean = v_hat_sum / n_clients

        NMSE += tf.reduce_sum((vec - v_hat_mean) ** 2) / tf.reduce_sum(vec ** 2)
    end = time.time()

    print(f'elapsed time {end - start}')
    print(f'the average NMSE is {NMSE/n_trials}')


if __name__ == '__main__':
    test_dme(1, 1_000_000, 10_000_000, 1000)