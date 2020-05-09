"""Utility functions.

This module contains many jitted functions, mainly for speed up in simulation and regression analysis.
"""
from itertools import product
import math

import numba as nb
import numpy as np
from scipy.interpolate import UnivariateSpline
import pandas as pd
import matplotlib.pyplot as plt

import cbsp

# ---------- SIMULATION

@nb.jit(nopython=True)
def spike_train(size, freq, dt):
    """Poisson spike train.
    
    Args:
        size (int): length of the spike train. Either population size or time steps.
        freq (float): firing rate.
        dt (float): timestep.
    
    Returns:
        array: spikes, 0 or 1s, of length size.
    """
    out = np.zeros(size)
    for i in range(size):
        out[i] = 1 if np.random.sample() < freq * dt else 0
    return out


@nb.jit(nopython=True)
def heaviside(x):
    """
    Heaviside-function.
    """
    return 0.5 * (np.sign(x) + 1)


@nb.jit(nopython=True)
def standard_normal(size, dt):
    """Samples from a standard normal distribution.
    
    Args:
        size (int): resulting size of the array.
        dt (float): timestep. The standard deviation is 1/np.sqrt(dt).
    
    Returns:
        array: np.random.normal(0, 1/np.sqrt(dt), size)
    """
    return np.random.normal(0, 1/np.sqrt(dt), size)


def alphafilter(time, trace, width=0.1):
    """Applies a valid convolution of the alphafilter to a trace.
    
    Args:
        time (array): time.
        trace (array): trace.
        width (float, optional): determines the filter width=1/alpha. Defaults to 0.1.
    
    Returns:
        ma.array: filtered trace in a masked array.
    """

    def _alphafilter():
        alpha = 1 / width
        filt = alpha ** 2 * time * np.exp( - alpha * time)
        filt = filt[filt>=0]
        # we cut off the tail of the filter at 1/3 of the time
        return filt[:len(time)//3]

    filt = _alphafilter()
    convolved = np.convolve(trace, filt, mode='valid')
    N = len(time) // 3
    mask = np.ones_like(time)
    mask[N//2:len(convolved)+N//2] = 0
    data = np.zeros_like(time)
    data[N//2:len(convolved)+N//2] = convolved
    return np.ma.array(data, mask=mask)


def moving_average(time, trace, width=0.1, spikes=False):
    """Applies a valid convolution of a box to a trace.
    
    Args:
        time (array): time.
        trace (array): trace.
        width (float, optional): determines the filter width. Defaults to 0.1.
    
    Returns:
        ma.array: filtered trace in a masked array.
    """
    dt = time[1] - time[0]
    N = int(width / dt)
    assert N < len(trace)
    convolved = np.convolve(trace, np.ones(N)/N, mode='same')
    if spikes:
        return convolved / dt
    return convolved
    # mask = np.ones_like(time)
    # mask[N//2:len(convolved)+N//2] = 0
    # data = np.zeros_like(time)
    # data[N//2:len(convolved)+N//2] = convolved
    # return np.ma.array(data, mask=mask)


def trace_pop_mean_with_std(time, trace, fig=None, ax=None, figsize=[5, 5], **kw):
    """Plots the mean trace on top of it's standard deviation region.
    
    Args:
        time (array): time.
        trace (array): trace of shape (#samples, #timesteps).
        fig (mpl.Figure, optional): matplotlib figure object. Defaults to None.
        ax (mpl.Axes, optional): matplotlib ax object. Defaults to None.
        figsize (list, optional): figure size in inches. Defaults to [5, 5].
    
    Returns:
        tuple: fig and ax objects.
    """
    fig = fig or plt.figure(figsize=figsize)
    ax = ax or plt.subplot()
    yerr = np.std(trace, axis=0)
    trace = np.mean(trace, axis=0)
    ax.plot(time, trace, **kw)
    kw.pop('label', None)
    ax.fill_between(time, trace-yerr, trace+yerr, alpha=0.3, **kw)
    return fig, ax


def sim_rbp(u, v, w0, estimator, coefs):
    """Simulates rate-based synaptic plasticity.

    Args:
        u (array): recorded presynaptic spikes of shape (#synapses, #timesteps).
        v (array): recorded postsynaptic spikes of shape (#timesteps).
        w0 (float): initial synaptic strength.
        estimator (Tuple[str]): estimator, e.g. ('u*v', 'u*v*w', 'u*w**2').
        coefs (array): sorted coefficients for features in estimator.

    Returns:
        array: trace of the synaptic strength.
    """
    time = np.linspace(0., cbsp.SIMULATION_TIME, int(cbsp.SIMULATION_TIME / cbsp.TIMESTEP) + 1)
    coefs = np.array(coefs).astype(str)

    assert len(coefs) == len(estimator)

    def _step(u, v, w):
        formula = ''
        for i, feature in enumerate(estimator):
            formula += '+' + coefs[i] + '*' + feature
        return eval(formula)

    u = cbsp.utils.moving_average(time, u.mean(axis=0), width=0.100, spikes=True)
    v = cbsp.utils.moving_average(time, v, width=0.500, spikes=True)

    w = np.zeros_like(time)
    w[0] = w0
    for i in range(len(time) - 1):
        w[i+1] = w[i] + cbsp.TIMESTEP * _step(u[i], v[i], w[i])

    return w


# ----------- STDP to RBP

def derivative(weights, time):
    """Takes the derivative of the average STDP at time 0.

    Args:
        weights (array): average STDP trace.
        time (array): time.
    
    Returns:
        float: the population average change of synapse strength at time point 0.
    """
    f = UnivariateSpline(time, weights, k = 1, s = 0.1)  # Note s here!
    dfdt = f.derivative()
    return dfdt(0)


# ----------- GET WEIGHTS

def get_weights(rbp):
    """Calculates weights as inverse variances.

    Args:
        rbp (array): rate-based plasticity of shape (#random_states, *)
    
    Returns:
        array: weights.
    """
    std = np.std(rbp, axis = 0)
    std[std<1e-5] = 1e-5
    weights = 1/std**2
    return weights


# ----------- FEATURE MATRICES

featstr = np.array(['1', 'u', 'v', 'w', 'u**2', 'u*v', 'u*w', 'v**2', 'v*w', 'w**2',
                    'u**2*v', 'u**2*w', 'u*v**2', 'u*v*w', 'u*w**2', 'v**2*w', 'v*w**2',
                    'u**2*v**2', 'u**2*v*w', 'u**2*w**2', 'u*v**2*w', 'u*v*w**2',
                    'v**2*w**2', 'u**2*v**2*w', 'u**2*v*w**2', 'u*v**2*w**2',
                    'u**2*v**2*w**2'])


def feature_matrix_p1(u, v, w):
    """Feature matrix for population 1 as DataFrame.
    
    Args:
        u (array): presynaptic firing rates.
        v (array): postsynaptic firing rates.
        w (array): synapse strengths.
    
    Returns:
        pd.DataFrame: feature matrix.
    """
    comb = np.array(list(product(u, v, w)))
    u = comb[:, 0]
    v = comb[:, 1]
    w = comb[:, 2]
    X = pd.DataFrame()  # np.zeros(u.size*v.size*w.size, featstr.size)
    for i, feat in enumerate(featstr, 1):
        X[feat] = eval(feat)
    X['1'] = np.ones_like(u)
    return X


def feature_matrix_p2(u, v, w):
    """Feature matrix for population 2 as DataFrame.
    See ~cbsp.utils.feature_matrix_p1(u, v, w).
    """
    comb = np.array(list(product(u, w)))
    u = comb[:, 0]
    w = comb[:, 1]
    X = pd.DataFrame()  # np.zeros(u.size*v.size*w.size, featstr.size)
    for i, feat in enumerate(featstr, 1):
        X[feat] = eval(feat)
    X['1'] = np.ones_like(u)
    return X


def feature_matrix_p3(u, v, w):
    """Feature matrix for population 3 as DataFrame.
    See ~cbsp.utils.feature_matrix_p1(u, v, w).
    """
    comb = np.array(list(product(u, w, u, w)))
    u = comb[:, 0]
    w = comb[:, 1]
    X = pd.DataFrame()  # np.zeros(u.size*v.size*w.size, featstr.size)
    for i, feat in enumerate(featstr, 1):
        X[feat] = eval(feat)
    X['1'] = np.ones_like(u)
    return X

# ----------- WEIGHTED LEAST SQUARES CROSSVALIDATION

@nb.jit(nopython=True, parallel=False) # is faster on a single thread
def crossvalidate(X, y, weights, splits, alpha=0, use_weights_for_r2=True, use_adj_r2=True):
    """Crossvalidation routine using weighted (regularized) least squares.
    
    Args:
        X (array): feature matrix of shape (#samples, #features).
        y (array): target of shape #samples.
        weights (array): weights of shape #samples.
        splits (int): number of validation splits.
        alpha (int, optional): regularization parameter. Defaults to 0.
        use_weights_for_r2 (bool, optional): Evaluate R in the weighted feature space. Defaults to True.
        use_adj_r2 (bool, optional): Adjust R with respect to number of features. Defaults to True.
    
    Returns:
        tuple: (r, coefs, coefs_std)
    """
    np.random.seed(99)  # for quantitative reproducibility
    n_obs = len(y)
    n_kfold = math.floor(n_obs / splits)
    indices = np.arange(0, n_obs)
    r2s = np.zeros(splits)
    coefs = np.zeros((splits, X.shape[1]))
    for i in range(splits):
        test_index = np.random.choice(indices, size=n_kfold, replace=False)
        test_mask = np.zeros(n_obs, dtype=np.bool_)
        test_mask[test_index] = True
        X_train, X_test = X[~test_mask], X[test_mask]
        y_train, y_test = y[~test_mask], y[test_mask]
        w_train, w_test = weights[~test_mask], weights[test_mask]
        b = wls(X_train, y_train, w_train, alpha=alpha)
        coefs[i] = b
        y_est = X_test @ b
        r2 = r2_score(y_test, y_est, w_test, X_test, use_weights=use_weights_for_r2, adjust=use_adj_r2)
        r2s[i] = r2
    meanr2 = np.mean(r2s)
    return meanr2, mean_2d(coefs), np.sqrt(var_2d(coefs))


@nb.jit(nopython=True)
def r2_score(y, y_est, weights, X, use_weights=True, adjust=True):
    """Goodness of fit.
    
    Args:
        y (array): target of shape #samples
        y_est (array): prediction of shape #samples.
        weights (array): weights of shape #samples.
        X (array): feature matrix of shape #samples, #features.
        use_weights (bool, optional): Evaluate R in the weighted feature space. Defaults to True.
        use_adj (bool, optional): Adjust R with respect to number of features. Defaults to True.
    
    Returns:
        float: measure of fit.
    """
    if not use_weights:
        weights = np.ones(y.size)
    numerator = (weights * (y - y_est) ** 2).sum()
    y_weighted_mean = (weights * y).sum() / weights.sum()
    denominator = (weights * (y - y_weighted_mean) ** 2).sum()
    r2 = 1 - numerator / denominator
    if adjust:
        r2 = adjust_r2(r2, X)
    return r2


@nb.jit(nopython=True)
def adjust_r2(r2, X):
    """
    Adjusts by the number of features (ref. Theil 1958).
    """
    return r2 - (1 - r2) * (X.shape[1] - 1) / (X.shape[0] - X.shape[1] - 1)


@nb.jit(nopython=True)
def _repeat(weights, X):
    p = X.shape[1]
    out = np.zeros((p, len(weights)))
    for i in range(p):
        out[i] = weights
    return out.T


@nb.jit(nopython=True)
def weight_sample(X, y, weights):
    """
    Transforms the sample into the weighted space.
    """
    weights = np.sqrt(weights)
    W = _repeat(weights, X) # repeat for cheaper elementwise multiplication
    return X*W, y*weights


@nb.jit(nopython=True)
def wls(X, y, weights, alpha=0):
    """Weighted least squares.
    
    Args:
        X (array): feature matrix of shape #samples, #features.
        y (array): target of shape #samples.
        weights (array): weight of shape #samples.
        alpha (int, optional): regularization parameter. Defaults to 0.
    
    Returns:
        array: coefficients of shape #features.
    """
    X_weighted, y_weighted = weight_sample(X, y, weights)
    I = np.eye(X.shape[1])
    b = np.linalg.inv(X_weighted.T @ X_weighted + alpha * I) @ (X_weighted.T @ y_weighted)
    return b


@nb.jit(nopython=True)
def mean_2d(x, axis=0):
    """
    Mean over axis of a 2d array as specifying an axis in numba is not supported. 
    """
    if axis==0:
        x = x.T
    mean = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        mean[i] = x[i].mean()
    return mean


@nb.jit(nopython=True)
def var_2d(x, axis=0):
    """
    Variance over axis of a 2d array as specifying an axis in numba is not supported. 
    """
    if axis == 0:
        x = x.T
    var = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        var[i] = x[i].var()
    return var

# ----------- UNIFIED ESTIMATOR

def unified_estimator(es_p1, es_p2, es_p3):
    """Determines the unified estimators ranking.
    
    Args:
        es_p1 (ExhaustiveSearch): exhaustive search object fitted to the data of population 1.
        es_p2 (ExhaustiveSearch): exhaustive search object fitted to the data of population 1.
        es_p3 (ExhaustiveSearch): exhaustive search object fitted to the data of population 1.
        
    Returns:
        tuple: index, estimators
            with 
                index (List[int])
                estimator (List[str])
    """
    index = sorted(es_p1.rs)
    unified_accuracy = {}
    
    for i in index:
        _rs = np.array([es_p1.rs[i], es_p2.rs[i], es_p3.rs[i]])
        unified_accuracy[i] = np.mean(_rs) - np.std(_rs)
        
    unified_accuracy = dict(sorted(unified_accuracy.items(),
                       key=lambda kv: (kv[1], kv[0]),
                       reverse=True))
    unified_estimator = {i: es_p1.estimators[i] for i in unified_accuracy}
    return list(unified_estimator.keys()), list(unified_estimator.values())