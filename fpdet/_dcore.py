# -*- coding: utf-8 -*-
"""
Created on Fri Oct 02 08:39:31 2019

@author: Pavel Gostev
@email: gostev.pavel@physics.msu.ru

Версия алгоритма расчета статистики фотоотсчетов
на основе numpy.float128
"""

import numpy as np

from ._dmatrix import d_binomial, d_subbinomial
from ._dmatrix import invd_binomial, invd_subbinomial


def d_matrix(qe: float, N: int, M: int, mtype: str = 'binomial', n_cells: int = 0) -> np.matrix:
    """
    Method for construction of binomial or subbinomial photodetection matrix
    with size NxM

    Parameters
    ----------
    qe : float
        Photon Detection Efficiency (PDE) of the detector.
    N : int
        Maximum numbers of photons
    M : int
        Maximum number of photocounts
    mtype : str, {'binomial', 'subbinomial'}, optional
        Type of the detector: ideal is binomial, realistic is subbinomial,
        but in the most of applications one can consider the detector as binomial
        The default is 'binomial'.
    n_cells : int, optional
        Number of photocounting cells in the subbinomial case. The default is 0.

    Raises
    ------
    ValueError
        Wrong method for the matrix construction.
        mtype must be binomial or subbinomial.

    Returns
    -------
    numpy.ndarray
        Binomial or subbinomial photodetection matrix of size NxM.

    """

    if mtype == 'binomial':
        return d_binomial(qe, N, M)
    elif mtype == 'subbinomial':
        return d_subbinomial(qe, N, M, n_cells)
    else:
        raise ValueError(
            f"Can't construct detection matrix of type {mtype}. mtype must be 'binomial' or 'subbinomial'")


def invd_matrix(qe: float, N: int, M: int, mtype: str = 'binomial', n_cells: int = 0) -> np.matrix:
    """
    Method for construction of binomial or
    subbinomial inverse photodetection matrix
    with size MxN

    Parameters
    ----------
    qe : float
        Photon Detection Efficiency (PDE) of the detector.
    N : int
        Maximum numbers of photons
    M : int
        Maximum number of photocounts
    mtype : {'binomial', 'subbinomial'}, optional
        Type of the detector: ideal is binomial, realistic is subbinomial,
        but in most applications one can consider the detector as binomial
        The default is 'binomial'.
    n_cells : int, optional
        Number of photocounting cells is subbinomial case. The default is 0.

    Raises
    ------
    ValueError
        Wrong method for the matrix construction.
        mtype must be binomial or subbinomial.

    Returns
    -------
    numpy.ndarray
        Binomial or subbinomial inverse photodetection matrix of size MxN.

    """
    if mtype == 'binomial':
        return invd_binomial(qe, N, M)
    elif mtype == 'subbinomial':
        return invd_subbinomial(qe, N, M, n_cells)
    else:
        raise ValueError(
            f"Can't construct inversed detection matrix of type {mtype} mtype must be binomial or subbinomial")


def P2Q(P: np.ndarray, qe: float, M: int = 0, mtype: str = 'binomial', n_cells: int = 0) -> np.ndarray:
    """
    Method for calculation of photocounting statistics
    from photon-number statistics

    Parameters
    ----------
    P : numpy.ndarray
        Photon-number statistics.
    qe : float
        Photon Detection Efficiency (PDE) of the detector.
    M : int, optional
        Maximum number of photocounts. It's undependent from length of P
        The default is 0, and in this case length(Q) = length(P).
    mtype : {'binomial', 'subbinomial'}, optional
        Type of the detector: ideal is binomial, realistic is subbinomial,
        but in most applications one can consider the detector as binomial
        The default is 'binomial'.
    n_cells : int, optional
        Number of photocounting cells is subbinomial case. The default is 0.

    Returns
    -------
    Q : numpy.ndarray
        Photocounting statistics.

    """

    N = len(P)
    if M == 0:
        M = N
    return d_matrix(qe, N, M, mtype, n_cells).dot(P)


def Q2P(Q: np.ndarray, qe: float, N: int = 0, mtype: str = 'binomial', n_cells: int = 0) -> np.ndarray:
    """
    Method for calculation of photon-number statistics
    from photocounting statistics by simple inversion

    Parameters
    ----------
    Q : numpy.ndarray
        Photocounting statistics.
    qe : float
        Photon Detection Efficiency (PDE) of the detector.
    N : int, optional
        Maximum number of photons. It's undependent from length of Q
        The default is 0, and in this case length(Q) = length(P).
    mtype : {'binomial', 'subbinomial'}, optional
        Type of the detector: ideal is binomial, realistic is subbinomial,
        but in most applications one can consider the detector as binomial
        The default is 'binomial'.
    n_cells : int, optional
        Number of photocounting cells is subbinomial case. The default is 0.

    Returns
    -------
    P : numpy.ndarray
        Photon-number statistics.

    """
    M = len(Q)
    if N == 0:
        N = M
    if N > M:
        Q = np.concatenate((Q, np.zeros(N - M)))
        M = N

    return invd_matrix(qe, M, N, mtype, n_cells).dot(Q)
