from __future__ import absolute_import, division, print_function

from functools import partial

import numpy as np
from scipy.spatial import KDTree,minkowski_distance
from scipy.special import psi
from joblib import Parallel, delayed


def estimate(X, Y, k=None, n_jobs=1):
    """ Estimate univesal k-NN divergence.

    Parameters
    ----------
        X, Y:
            2-dimensional array where each row is a sample.

        k:
            k-NN to be used. None for adaptive choice.

        n_jobs:
            number of jobs to run in parallel. Python 2 may only work with
            ``n_jobs=1``.
    """

    if not (isinstance(k, int) or k is None):
        raise ValueError('k has incorrect type.')
    if k is not None and k <= 0:
        raise ValueError('k cannot be <= 0')
    X = np.array(X)
    Y = np.array(Y)
    if len(X.shape) != 2 or len(Y.shape) != 2:
        raise ValueError('X or Y has incorrect dimension.')
    if X.shape[0] <= 1 or Y.shape[0] <= 1:
        raise ValueError('number of samples is not sufficient.')
    if X.shape[1] != Y.shape[1]:
        raise ValueError('numbers of columns of X and Y are different.')
    d = X.shape[1]
    n = X.shape[0]
    m = Y.shape[0]

    X_tree = KDTree(X)
    Y_tree = KDTree(Y)

    P = Parallel(n_jobs)
    nhu_ro = P(delayed(__calc_nu_rho_first_term)(x, X_tree, Y_tree, k) for x in X)
    r = (d / n) * sum(nhu_ro) + np.log(m / (n - 1))
    if k is None:
        l_k = P(delayed(__calc_l_k)(x, X_tree, Y_tree) for x in X)
        r += (1 / n) * sum(l_k)
    return r

def getdistance(dataPoint,tree):
    rho_d_list = []
    for data in tree.data:
        if not (data == np.array(dataPoint)).all():
            rho_d_list.append(minkowski_distance(np.array(dataPoint), data))
    return min(rho_d_list)

def __get_epsilon(a, X_tree, Y_tree):
    rho_d = getdistance(a,X_tree)
    nu_d = getdistance(a,Y_tree)
    return rho_d,nu_d


def __get_epsilon_sample_num(a, tree, X_tree, Y_tree, default_offset=0):
    e = __get_epsilon(a, X_tree, Y_tree)
    e = max(e[0], e[1]) + 0.5 ** 40
    return len(tree.query_ball_point(a, e)) - default_offset



def __get_distance(a, tree, X_tree, Y_tree, k, default_offset):
    if k is None:
        k_ = __get_epsilon_sample_num(a, tree, X_tree, Y_tree)
    else:
        k_ = k + default_offset
    d, _ = tree.query([a], k_)
    return d[0] if k_ == 1 else d[0][-1]


def __get_distance_new(a, tree, X_tree, Y_tree,):
    dist  = __get_epsilon(a, X_tree, Y_tree)
    return dist

def __calc_nu_rho(x, X_tree, Y_tree, k):
    rho = partial(__get_distance, tree=X_tree, default_offset=1,
                  X_tree=X_tree, Y_tree=Y_tree, k=k)
    nu = partial(__get_distance, tree=Y_tree, default_offset=0,
                 X_tree=X_tree, Y_tree=Y_tree, k=k)
    return np.log(nu(x) / rho(x))

def __calc_nu_rho_first_term(x, X_tree, Y_tree, k):
    distFn = partial(__get_distance_new, tree=X_tree,
                  X_tree=X_tree, Y_tree=Y_tree)
    rho_d, nu_d = distFn(x)
    rho_d = rho_d + 0.5 ** 40
    nu_d = nu_d + 0.5 ** 40
    # nu = partial(__get_distance, tree=Y_tree, default_offset=0,
    #              X_tree=X_tree, Y_tree=Y_tree, k=k)
    return np.log(nu_d / rho_d)

def __calc_l_k(x, X_tree, Y_tree):
    _l = partial(__get_epsilon_sample_num, tree=X_tree, default_offset=1,
                 X_tree=X_tree, Y_tree=Y_tree)
    _k = partial(__get_epsilon_sample_num, tree=Y_tree, default_offset=0,
                 X_tree=X_tree, Y_tree=Y_tree)
    return psi(_l(x)) - psi(_k(x))