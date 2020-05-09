"""Simulations of neural population 2.

Simulations of linear- and non-linear, calcium-based, spike-timing-dependent synaptic plasticity 
of a homogeneous presynaptic population of 1000 neurons wired onto a single postsynaptic neuron.
Postsynaptic firing underlies the MAT or AEIF model.
Methods for abstracting the STDP to rate-based plasticity for large parameter spaces.

    Simple usage example:

        cbsp.set_simulation_time(2.0)
        cbsp.set_timstep(0.001)
        cbsp.population_2.linear_calcium_mat(u=10, w0=0.5, seed=10)
"""
import os
import multiprocessing
from itertools import product
from tqdm.auto import tqdm

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize

import cbsp
from cbsp import utils
from cbsp.utils import feature_matrix_p2 as feature_matrix

# ---------------- STDP

@nb.jit(nopython=True)
def linear_calcium_mat(u, w0, seed):
    """Integrates the spike-timing dependent synaptic strength.
    
    Args:
        u (float): presynaptic firing rate.
        w0 (float): initial synapse strength.
        seed (int): random state.
    
    Returns:
        tuple: (w, t, (u, v, I, c))
            with 
                array: w, change of synapse strengths. Shape (#synapses, #timesteps).
                array: t, time.
                array: u, presynaptic spike trains. Shape (#synapses, #timesteps).
                array: v, postsynaptic spike train. Shape (#timesteps).
                array: I, postsynaptic current. Shape (#timesteps).
                array: c, calcium traces. Shape (#synapses, #timesteps).
    """
    # Calcium
    tau_Ca = 0.02227212
    Cpre = 0.84410
    Cpost = 1.62138

    # Plasticity
    thetaD = 1
    thetaP = 2.009289
    drate = 137.7586
    prate = 597.08922
    sigma = 2.8284
    tau = 520.76129
    sqrttau = np.sqrt(tau)

    # MAT model
    tau_m, tau_1, tau_2 = 5.e-3, 10.e-3, 200.e-3
    trefr = 2.e-3
    c_N = 0.4
    R = 50e6
    alpha1, alpha2, w_mat = 30.e-3, 2.e-3, 20.e-3
    Iconst = c_N * tau_m / R

    time = np.linspace(0., cbsp.SIMULATION_TIME, int(cbsp.SIMULATION_TIME / cbsp.TIMESTEP) + 1)
    N = 1000
    V = 0.
    theta1, theta2 = 0., 0.
    Theta = 0.
    trest = 0.
    I = 0.
    v_sp = 0.

    np.random.seed(seed)
    c = np.zeros(N)
    w = np.zeros(N)
    w[:] = w0  # np.random.normal(w0, w0_std, N)
    w_rec_pop = np.zeros((N, len(time)))
    u_rec_pop = np.zeros((N, len(time)))
    v_rec = np.zeros_like(time)
    I_rec = np.zeros_like(time)
    c_rec_pop = np.zeros((N, len(time)))

    for i, t in enumerate(time):
        u_sp = utils.spike_train(N, u, cbsp.TIMESTEP)
        n = utils.standard_normal(N, cbsp.TIMESTEP)
        Hp = utils.heaviside(c - thetaP)
        Hd = utils.heaviside(c - thetaD)
        c = c - cbsp.TIMESTEP * c / tau_Ca + Cpre * u_sp + Cpost * v_sp
        w = w + cbsp.TIMESTEP / tau * (prate * (1 - w) * Hp - drate * w * Hd + np.sqrt(Hp + Hd) * sigma * sqrttau * n)
        I = Iconst * np.dot(w, u_sp)

        V = V + cbsp.TIMESTEP * (-V / tau_m + R / tau_m * I)
        theta1 = theta1 + cbsp.TIMESTEP * (-theta1 / tau_1) + alpha1 * v_sp
        theta2 = theta2 + cbsp.TIMESTEP * (-theta2 / tau_2) + alpha2 * v_sp
        Theta = theta1 + theta2 + w_mat

        if V > Theta and t > trest:
            v_sp = 1.
            trest = t + trefr
        else:
            v_sp = 0.

        w_rec_pop[:, i] = w
        u_rec_pop[:, i] = u_sp
        v_rec[i] = v_sp
        I_rec[i] = I
        c_rec_pop[:, i] = c

    return w_rec_pop, time, (u_rec_pop, v_rec, I_rec, c_rec_pop)


@nb.jit(nopython=True)
def non_linear_calcium_mat(u, w0, seed):
    """
    Same as ~cbsp.population_2.linear_calcium_mat(u, w0, seed) for the non linear calcium model.
    """
    # Calcium
    tau_Ca = 0.01893044
    Cpre = 0.86467
    Cpost = 2.30815
    xi = (2 * (Cpost + Cpre) - Cpost) / Cpre - 1

    # Plasticity
    thetaD = 1
    thetaP = 4.99780
    drate = 111.82515
    prate = 894.23695
    sigma = 2.8284
    tau = 707.02258
    sqrttau = np.sqrt(tau)

    # MAT model
    tau_m, tau_1, tau_2 = 5.e-3, 10.e-3, 200.e-3
    trefr = 2.e-3
    c_N = 0.4
    R = 50e6
    alpha1, alpha2, w_mat = 30.e-3, 2.e-3, 20.e-3
    Iconst = c_N * tau_m / R

    time = np.linspace(0., cbsp.SIMULATION_TIME, int(cbsp.SIMULATION_TIME / cbsp.TIMESTEP) + 1)
    N = 1000
    V = 0.
    theta1, theta2 = 0., 0.
    Theta = 0.
    trest = 0.
    I = 0.
    v_sp = 0.

    np.random.seed(seed)

    cpre = np.zeros(N)
    cpost = np.zeros(N)
    c = np.zeros(N)
    w = np.zeros(N)
    w[:] = w0  # np.random.normal(w0, w0_std, N)
    w_rec_pop = np.zeros((N, len(time)))
    u_rec_pop = np.zeros((N, len(time)))
    v_rec = np.zeros_like(time)
    I_rec = np.zeros_like(time)
    c_rec_pop = np.zeros((N, len(time)))

    for i, t in enumerate(time):
        u_sp = utils.spike_train(N, u, cbsp.TIMESTEP)
        n = utils.standard_normal(N, cbsp.TIMESTEP)
        Hp = utils.heaviside(c - thetaP)
        Hd = utils.heaviside(c - thetaD)
        cpre = cpre - cbsp.TIMESTEP * cpre / tau_Ca + Cpre * u_sp
        cpost = cpost - cbsp.TIMESTEP * cpost / tau_Ca + Cpost * v_sp + xi * v_sp * cpre
        c = cpre + cpost
        w = w + cbsp.TIMESTEP / tau * (prate * (1 - w) * Hp - drate * w * Hd + np.sqrt(Hp + Hd) * sigma * sqrttau * n)
        I = Iconst * np.dot(w, u_sp)

        V = V + cbsp.TIMESTEP * (-V / tau_m + R / tau_m * I)
        theta1 = theta1 + cbsp.TIMESTEP * (-theta1 / tau_1) + alpha1 * v_sp
        theta2 = theta2 + cbsp.TIMESTEP * (-theta2 / tau_2) + alpha2 * v_sp
        Theta = theta1 + theta2 + w_mat

        if V > Theta and t > trest:
            v_sp = 1.
            trest = t + trefr
        else:
            v_sp = 0.

        w_rec_pop[:, i] = w
        u_rec_pop[:, i] = u_sp
        v_rec[i] = v_sp
        I_rec[i] = I
        c_rec_pop[:, i] = c

    return w_rec_pop, time, (u_rec_pop, v_rec, I_rec, c_rec_pop)


@nb.jit(nopython=True)
def linear_calcium_aeif(u, w0, seed):
    """
    Same as ~cbsp.population_2.linear_calcium_mat(u, w0, seed) for the aeif model.
    """
    # Calcium
    tau_Ca = 0.02227212
    Cpre = 0.84410
    Cpost = 1.62138

    # Plasticity
    thetaD = 1
    thetaP = 2.009289
    drate = 137.7586
    prate = 597.08922
    sigma = 2.8284
    tau = 520.76129
    sqrttau = np.sqrt(tau)

    # AEIF model (ranamed w->z)
    C = 2.81e-10 # 2.81e-9  # pF
    g_L = 30e-9  # nS
    E_L = -70.6e-3  # mV
    V_T = -50.4e-3  # mV
    D_T = 2e-3  # mV
    tau_z = 0.144  # s
    a = 4e-9  # nS
    b = 0.0805e-9  # nA
    R = 1 / g_L
    tau_v = R * C
    c_N = 0.17 # 0.075

    Iconst = c_N * tau_v / R

    time = np.linspace(0., cbsp.SIMULATION_TIME, int(cbsp.SIMULATION_TIME / cbsp.TIMESTEP) + 1)
    N = 1000
    V = E_L
    z = 0.
    I = 0.
    v_sp = 0.

    np.random.seed(seed)

    c = np.zeros(N)
    w = np.zeros(N)
    w[:] = w0 # np.random.normal(w0, w0_std, N)
    w_rec = np.zeros_like(time)
    v_rec = np.zeros_like(time)
    w_rec_pop = np.zeros((N, len(time)))
    u_rec_pop = np.zeros((N, len(time)))
    v_rec = np.zeros_like(time)
    I_rec = np.zeros_like(time)
    c_rec_pop = np.zeros((N, len(time)))

    for i, t in enumerate(time):
        # import pdb; pdb.set_trace()
        u_sp = utils.spike_train(N, u, cbsp.TIMESTEP)
        n = utils.standard_normal(N, cbsp.TIMESTEP)
        Hp = utils.heaviside(c - thetaP)
        Hd = utils.heaviside(c - thetaD)
        c = c - cbsp.TIMESTEP * c / tau_Ca + Cpre * u_sp + Cpost * v_sp
        w = w + cbsp.TIMESTEP / tau * (prate * (1 - w) * Hp - drate * w * Hd + np.sqrt(Hp + Hd) * sigma * sqrttau * n)
        I = Iconst * np.dot(w, u_sp)

        V = V + cbsp.TIMESTEP / tau_v * (- V + E_L + D_T * np.exp((V - V_T) / D_T) - R * z + R * I)
        z = z + cbsp.TIMESTEP / tau_z * (a * (V - E_L) - z)

        if V > V_T:
            v_sp = 1.
            V = V - (V_T - E_L)
            z = z + b
        else:
            v_sp = 0.

        w_rec_pop[:, i] = w
        u_rec_pop[:, i] = u_sp
        v_rec[i] = v_sp
        I_rec[i] = I
        c_rec_pop[:, i] = c

    return w_rec_pop, time, (u_rec_pop, v_rec, I_rec, c_rec_pop)

# ---------------- RBP

def stdp2rbp_linear_calcium_mat(*args, **kwargs):
    """Rate-based plasticity from STDP using the linear calcium and MAT model.
    
    Args:
        - same as cbsp.population_2.linear_calcium_mat(u, v, w0, seed)

    Returns:
        tuple: (w, v)
            with
                float: w, the population average change of synapse strength at time point 0
                float: v, the postsynaptic firing rate within the first 500ms.
    """
    w_rec, t, (_, v_rec, _, _) = linear_calcium_mat(*args, **kwargs)
    return utils.derivative(w_rec.mean(axis=0), t), v_rec[0:int(0.5 / cbsp.TIMESTEP)].sum() * 2


def stdp2rbp_non_linear_calcium_mat(*args, **kwargs):
    """Rate-based plasticity from STDP using the non linear calcium and MAT model.
    
    Args:
        - same as cbsp.population_2.non_linear_calcium_mat(u, v, w0, seed)

    Returns:
        tuple: (w, v)
            with
                float: w, the population average change of synapse strength at time point 0
                float: v, the postsynaptic firing rate within the first 500ms.
    """
    w_rec, t, (_, v_rec, _, _) = non_linear_calcium_mat(*args, **kwargs)
    return utils.derivative(w_rec.mean(axis=0), t), v_rec[0:int(0.5 / cbsp.TIMESTEP)].sum() * 2


def stdp2rbp_linear_calcium_aeif(*args, **kwargs):
    """Rate-based plasticity from STDP using the linear calcium and AEIF model.
    
    Args:
        - same as cbsp.population_2.linear_calcium_aeif(u, v, w0, seed)

    Returns:
        tuple: (w, v)
            with
                float: w, the population average change of synapse strength at time point 0
                float: v, the postsynaptic firing rate within the first 500ms.
    """
    w_rec, t, (_, v_rec, _, _) = linear_calcium_aeif(*args, **kwargs)
    return utils.derivative(w_rec.mean(axis=0), t), v_rec[0:int(0.5 / cbsp.TIMESTEP)].sum() * 2


def main_linear_calcium_mat(u=np.arange(0, 101),
                            w=np.arange(0, 1.05, 0.05),
                            seed=np.arange(0, 100),
                            nproc=2):
    """RBP from STDP for the whole parameter space using the linear calcium and MAT model.
    
    Args:
        u (array, optional): presynaptic firing rates. Defaults to np.arange(0, 101).
        w (array, optional): initial synaptic strengths. Defaults to np.arange(0, 1.05, 0.05).
        seed (array, optional): random states. Defaults to np.arange(0, 100).
        nproc (int, optional): number of processes to use. Defaults to 8.
    
    Returns:
        array: rate-based plasticity and postsynaptic firing for all possible combinations of u, and w.
                Has shape (#random_states, u.size * w.size, 2).
    """
    pool = multiprocessing.Pool(processes=nproc)
    results = np.zeros([seed.size, u.size*w.size, 2])
    for i, s in enumerate(tqdm(seed, desc='Seed')):
        results[i] = np.array(pool.starmap(stdp2rbp_linear_calcium_mat, product(u, w, np.array([s]))))
    return results


def main_non_linear_calcium_mat(u=np.arange(0, 101),
                            w=np.arange(0, 1.05, 0.05),
                            seed=np.arange(0, 100),
                            nproc=2):
    """
    Same as ~cbsp.population_2.main_linear_calcium_mat(u, w, seed, nproc) for the non linear calcium model.
    """
    pool = multiprocessing.Pool(processes=nproc)
    results = np.zeros([seed.size, u.size*w.size, 2])
    for i, s in enumerate(tqdm(seed, desc='Seed')):
        results[i] = np.array(pool.starmap(stdp2rbp_non_linear_calcium_mat, product(u, w, np.array([s]))))
    return results


def main_linear_calcium_aeif(u=np.arange(0, 101),
                            w=np.arange(0, 1.05, 0.05),
                            seed=np.arange(0, 100),
                            nproc=2):
    """
    Same as ~cbsp.population_2.main_linear_calcium_mat(u, w, seed, nproc) for the AEIF model.
    """
    pool = multiprocessing.Pool(processes=nproc)
    results = np.zeros([seed.size, u.size*w.size, 2])
    for i, s in enumerate(tqdm(seed)):
        results[i] = np.array(pool.starmap(stdp2rbp_linear_calcium_aeif, product(u, w, np.array([s]))))
    return results


def plot_3d(u, w, rbp_or_v, zlabel='\dot{w}'):
    """Plots the RBP or postsynaptic firing rate in 3d.
    
    Args:
        u (array): presynaptic firing rates.
        w (array): initial synapse strengths.
        rbp_or_v (array): rate-based plasticity or postsynaptic firing rate for the parameter space.
                     Flat or of shape (u.size, w.size).
        zlabel (str): zlabel function, '${zlabel}_(u, w)$'. Default is '\dot{w}'
    
    Returns:
        tuple: figure and axis objects.
    """
    class MidpointNormalize(Normalize):
        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            Normalize.__init__(self, vmin, vmax, clip)

        def __call__(self, value, clip=None):
            x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
            return np.ma.masked_array(np.interp(value, x, y))

    fig = plt.figure(figsize=[10, 5])
    U, W = np.meshgrid(u, w)
    Z = rbp_or_v.reshape(len(u), len(w))

    ax = fig.add_subplot(111, projection='3d')
    norm = MidpointNormalize(vmin=rbp_or_v.min()-1e-10, vmax=rbp_or_v.max(), midpoint=0)
    ax.plot_surface(X=U, Y=W, Z=Z.T, rstride=1, cstride=1, cmap=plt.cm.seismic, norm=norm, antialiased=False)
    ax.set_xlabel('$u$', fontsize = 10)
    ax.set_ylabel('$w$', fontsize = 10)
    zlabel = f'${zlabel}_(u, w)$'
    ax.set_zlabel(zlabel, fontsize=10)
    ax.grid(False)
    ax.view_init(11, 130)
    return fig, ax

plot_dynamics = cbsp.population_1.plot_dynamics  # the plot function works for both populations