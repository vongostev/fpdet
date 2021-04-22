#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 13:11:13 2020

@author: Pavel Gostev
@email: gostev.pavel@physics.msu.ru

numpy version
"""

import functools
import numpy as np
from scipy.special import factorial

__all__ = ['compose', 'lrange', 'fact', 'p_convolve', 'moment', 'mean', 'g2',
           'normalize', 'fidelity', 'entropy', 'DPREC']

try:
    DPREC = np.float128
except AttributeError:
    DPREC = np.float64
    print('%s:%s:' % (__file__, 19), 'RuntimeWarning:',
          "Numpy.float128 can not be used. DPREC is numpy.float64, results may be unprecise.")


def compose(*functions):
    """
    Utility function for the sequentually applying them on data

    >>> compose(mean, normalize)([1,1])
    0.5

    Parameters
    ----------
    *functions : number of functions

    Returns
    -------
    Function composition as function.

    """
    def pack(x): return x if type(x) is tuple else (x,)

    return functools.reduce(
        lambda acc, f: lambda *y: f(*pack(acc(*pack(y)))), reversed(functions), lambda *x: x)


def lrange(iterable):
    """
    Make np.arange with len identical to given iterable

    >>> lrange([1,1])
    array([0, 1])

    >>> lrange({'a': 1, 'b': 2})
    array([0, 1])

    Parameters
    ----------
    iterable : list, array, dict

    Returns
    -------
    numpy.array
        arange from 0 to len(iterable).

    """
    return compose(np.arange, len)(iterable)


def fact(n: int) -> int:
    """
    Alias to scipy.factorial

    Parameters
    ----------
    n : int

    Returns
    -------
    int

    """
    return DPREC(factorial(n, exact=True))


def p_convolve(p_signal, p_noise) -> np.ndarray:
    """
    Normalized convolution of two distributions
    >>> p_convolve([0.25, 0.25, 0.25, 0.25], [0.9, 0.1])
    array([0.23076923, 0.25641026, 0.25641026, 0.25641026])

    Parameters
    ----------
    p_signal : iterable
    p_noise : iterable

    Returns
    -------
    np.ndarray

    """
    N = max(len(p_signal), len(p_noise))
    return normalize(np.convolve(p_signal, p_noise)[:N])


def moment(p, n: int) -> float:
    """
    n-th initial moment of the given distribution

    Parameters
    ----------
    p : iterable
        Distribution.
    n : int
       Order of a moment.

    Returns
    -------
    float
        Moment value.

    """
    return sum(i ** n * p[i] for i in lrange(p))


def mean(p) -> float:
    """
    Mean value. Alias to moment(p, 1)

    Parameters
    ----------
    p : iterable
        Distribution.

    Returns
    -------
    float
        Mean value.

    """
    return moment(p, 1)


def g2(p) -> float:
    r"""
    Calculate quantum (normally-ordered) g2(0) value for the given distribution

    .. math:: g_2(0) = \frac{\langle n^2 \rangle - \langle n \rangle}{\langle n \rangle^2}

    Parameters
    ----------
    p : iterable
        Distribution.

    Returns
    -------
    float
        Quantum g2(0) value.

    """
    m = mean(p)
    s = moment(p, 2)
    if m == 0:
        m = 1
        np.warnings.warn_explicit(
            'Uncorrect g2. Zero mean change to 1', RuntimeWarning, 'fpdet._numpy_core.py', 168)
    return (s - m) / m ** 2


def normalize(p) -> np.ndarray:
    """
    Iterable normalization by sum and transformation to np.ndarray

    Parameters
    ----------
    p : iterable

    """
    return np.array(p) / moment(p, 0)


def fidelity(p, q) -> float:
    r"""
    Fidelity between two distributions.
    Distributions must have the same length.
    Formula modification allows to calculate 'fidelity' between distributions' estimations with negative elements:

    .. math:: F(p, q) = (\sum_ksign(p_kq_k)\sqrt{|p_kq_k|})^2

    Parameters
    ----------
    p : iterable

    q : iterable

    Raises
    ------
    ValueError
        'Distributions must have the same length, not {len(p1)}, {len(p2)}'.

    """
    if len(p) != len(q):
        raise ValueError(
            f'Distributions must have the same length, not {len(p)}, {len(q)}')
    prod = np.array(p) * np.array(q)
    return sum(np.sign(prod)*np.sqrt(np.abs(prod))) ** 2


def entropy(p) -> float:
    """
    Shannon entropy of the given distribution

    Parameters
    ----------
    p : iterable
        Distribution.

    Returns
    -------
    float
        Shannon entropy.

    """
    return sum(- e * np.log(e) for e in p if e > 0)
