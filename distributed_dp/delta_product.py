import math
from functools import lru_cache

from scipy.stats import norm

import tensorflow as tf

@lru_cache(maxsize=None)
def calc_expected_std_after_round(std=0.4):  # after Hadamard std ~~> l2/sqrt(d)
    p = float('inf')
    squared_coord = 0
    n = 1
    while p > 0:
        l, r = (n - 1 / 2), (n + 1 / 2)  # we safely skip the 0..0.5 interval since it rounds to zero
        probability_mass = norm.cdf(r, scale=std) - norm.cdf(l, scale=std)
        squared_coord += probability_mass * n ** 2
        n += 1
        p = norm.sf(r, scale=std)  # survival, which is more numerically stable 1 - cdf
    squared_coord *= 2
    # squared_coord is the second moment of the round function, i.e., E[Round(X)**2], for X ~ Normal(0, std**2)
    # (this is Round(X) std since E[X] is 0)
    return math.sqrt(squared_coord)


# @lru_cache(maxsize=None)
def calc_expected_round_product(std=0.4, prec=0):  # after Hadamard std ~~> l2/sqrt(d)
    # this calculates E(X*Round(X)) for X ~ N(0,std**2)

    p = float('inf')
    product = 0
    n = 1
    while p > prec:
        l, r = (n - 1 / 2), (n + 1 / 2)  # we safely skip the 0..0.5 interval since it rounds to zero
        e = norm.expect(lb=l, ub=r,
                        scale=std)  # this is the same as centroid * probability_mass (since centroid == e/probability_mass)
        product += e * n
        n += 1
        p = norm.sf(r, scale=std)  # survival, which is more numerically stable 1 - cdf
    return 2 * product  # multiply by 2 since we only covered positive values


@lru_cache(maxsize=None)
def scale_from_std(std):
    # given the constant above (denoted C) for a vector V ~ N(0, std**2)**d:
    # E(<V,round(V)>) => d * C
    # and EDEN/Drive scalar is l2(V)^2 / <V,round(V)> => (l2(V)^2 / d) / C => std**2 / C
    C = calc_expected_round_product(std)
    return std ** 2 / C


# @lru_cache(maxsize=None)
def scale_for_dp(quantize_scale, std, prec=0):
    # same as above, but considers the scale that distributed DP is using
    return tf.math.divide_no_nan(quantize_scale * std ** 2, calc_expected_round_product(quantize_scale * std, prec))
