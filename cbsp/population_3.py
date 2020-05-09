"""Simulations of neural population 3.

Simulations of linear- and non-linear, calcium-based, spike-timing-dependent synaptic plasticity 
of two independent homogeneous presynaptic population of 1000 neurons wired onto a single postsynaptic neuron.
Postsynaptic firing underlies the MAT or AEIF model.
Methods for abstracting the STDP to rate-based plasticity for large parameter spaces.

    Simple usage example:

        cbsp.set_simulation_time(2.0)
        cbsp.set_timstep(0.001)
        cbsp.population_3.linear_calcium_mat(u1=10, w1=0.5, u2=10, w2=10, seed=10)
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
from cbsp.utils import feature_matrix_p3 as feature_matrix

# ---------------- STDP

@nb.jit(nopython=True)
def linear_calcium_mat(u1, w1, u2, w2, seed):
    """Integrates the spike-timing dependent synaptic strength.
    
    Args:
        u1 (float): presynaptic firing rate of the first population.
        w1 (float): initial synapse strength of the first population.
        u2 (float): presynaptic firing rate of the second population.
        w2 (float): initial synapse strength of the second population.
        seed (int): random state.
    
    Returns:
        tuple: ((w1, w2), t, (u1, u2, v, I, c1, c2))
            with 
                array: w1, change of synapse strengths in the first population. Shape (#synapses, #timesteps).
                array: w2, change of synapse strengths in the second population. Shape (#synapses, #timesteps).
                array: t, time.
                array: u1, presynaptic spike trains in the first population. Shape (#synapses, #timesteps).
                array: u2, presynaptic spike trains in the second population. Shape (#synapses, #timesteps).
                array: v, postsynaptic spike train. Shape (#timesteps).
                array: I, postsynaptic current. Shape (#timesteps).
                array: c1, calcium traces in the first population. Shape (#synapses, #timesteps).
                array: c2, calcium traces in the second population. Shape (#synapses, #timesteps).
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
    c_N = 0.2
    R = 50e6
    alpha1, alpha2, w_mat = 30.e-3, 2.e-3, 20.e-3
    Iconst = c_N * tau_m / R

    time = np.linspace(0., cbsp.SIMULATION_TIME, int(cbsp.SIMULATION_TIME / cbsp.TIMESTEP) + 1)
    N1 = 1000
    N2 = 1000
    V = 0.
    theta1, theta2 = 0., 0.
    Theta = 0.
    trest = 0.
    I = 0.
    v_sp = 0.

    np.random.seed(seed)

    c = np.zeros(N1+N2)
    w = np.zeros(N1+N2)
    u_sp = np.zeros(N1+N2)
    w[:N1] = w1  # np.random.normal(w0, w0_std, N)
    w[N1:] = w2
    w1_rec_pop = np.zeros((N1, len(time)))
    w2_rec_pop = np.zeros((N2, len(time)))
    u1_rec_pop = np.zeros((N1, len(time)))
    u2_rec_pop = np.zeros((N2, len(time)))
    v_rec = np.zeros_like(time)
    I_rec = np.zeros_like(time)
    c1_rec_pop = np.zeros((N1, len(time)))
    c2_rec_pop = np.zeros((N2, len(time)))

    for i, t in enumerate(time):
        # import pdb; pdb.set_trace()
        u_sp[:N1] = utils.spike_train(N1, u1, cbsp.TIMESTEP)
        u_sp[N2:] = utils.spike_train(N2, u2, cbsp.TIMESTEP)
        n = utils.standard_normal(N1+N2, cbsp.TIMESTEP)
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

        w1_rec_pop[:, i] = w[:N1]
        u1_rec_pop[:, i] = u_sp[:N1]
        w2_rec_pop[:, i] = w[N1:]
        u2_rec_pop[:, i] = u_sp[N1:]
        v_rec[i] = v_sp
        I_rec[i] = I
        c1_rec_pop[:, i] = c[:N1]
        c2_rec_pop[:, i] = c[N1:]

    return (w1_rec_pop, w2_rec_pop), time, (u1_rec_pop, u2_rec_pop, v_rec, I_rec, c1_rec_pop, c2_rec_pop)


@nb.jit(nopython=True)
def non_linear_calcium_mat(u1, w1, u2, w2, seed):
    """
    Same as ~cbsp.population_3.linear_calcium_mat(u1, w1, u2, w2, seed) for the non linear calcium model.
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
    c_N = 0.2
    R = 50e6
    alpha1, alpha2, w_mat = 30.e-3, 2.e-3, 20.e-3
    Iconst = c_N * tau_m / R

    time = np.linspace(0., cbsp.SIMULATION_TIME, int(cbsp.SIMULATION_TIME / cbsp.TIMESTEP) + 1)
    N1 = 1000
    N2 = 1000
    V = 0.
    theta1, theta2 = 0., 0.
    Theta = 0.
    trest = 0.
    I = 0.
    v_sp = 0.

    np.random.seed(seed)

    cpre = np.zeros(N1+N2)
    cpost = np.zeros(N1+N2)
    c = np.zeros(N1+N2)
    w = np.zeros(N1+N2)
    u_sp = np.zeros(N1+N2)
    w[:N1] = w1  # np.random.normal(w0, w0_std, N)
    w[N1:] = w2
    w1_rec_pop = np.zeros((N1, len(time)))
    w2_rec_pop = np.zeros((N2, len(time)))
    u1_rec_pop = np.zeros((N1, len(time)))
    u2_rec_pop = np.zeros((N2, len(time)))
    v_rec = np.zeros_like(time)
    I_rec = np.zeros_like(time)
    c1_rec_pop = np.zeros((N1, len(time)))
    c2_rec_pop = np.zeros((N2, len(time)))

    for i, t in enumerate(time):
        # import pdb; pdb.set_trace()
        u_sp[:N1] = utils.spike_train(N1, u1, cbsp.TIMESTEP)
        u_sp[N2:] = utils.spike_train(N2, u2, cbsp.TIMESTEP)
        n = utils.standard_normal(N1+N2, cbsp.TIMESTEP)
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

        w1_rec_pop[:, i] = w[:N1]
        u1_rec_pop[:, i] = u_sp[:N1]
        w2_rec_pop[:, i] = w[N1:]
        u2_rec_pop[:, i] = u_sp[N1:]
        v_rec[i] = v_sp
        I_rec[i] = I
        c1_rec_pop[:, i] = c[:N1]
        c2_rec_pop[:, i] = c[N1:]

    return (w1_rec_pop, w2_rec_pop), time, (u1_rec_pop, u2_rec_pop, v_rec, I_rec, c1_rec_pop, c2_rec_pop)


@nb.jit(nopython=True)
def linear_calcium_aeif(u1, w1, u2, w2, seed):
    """
    Same as ~cbsp.population_3.linear_calcium_mat(u1, w1, u2, w2, seed) for the AEIF model.
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
    c_N = 0.085 # 17 # 0.0375

    Iconst = c_N * tau_v / R

    time = np.linspace(0., cbsp.SIMULATION_TIME, int(cbsp.SIMULATION_TIME / cbsp.TIMESTEP) + 1)
    N1 = 1000
    N2 = 1000
    V = E_L
    z = 0.
    I = 0.
    v_sp = 0.

    np.random.seed(seed)

    c = np.zeros(N1+N2)
    w = np.zeros(N1+N2)
    u_sp = np.zeros(N1+N2)
    w[:N1] = w1 # np.random.normal(w0, w0_std, N)
    w[N1:] = w2
    w1_rec_pop = np.zeros((N1, len(time)))
    w2_rec_pop = np.zeros((N2, len(time)))
    u1_rec_pop = np.zeros((N1, len(time)))
    u2_rec_pop = np.zeros((N2, len(time)))
    v_rec = np.zeros_like(time)
    I_rec = np.zeros_like(time)
    c1_rec_pop = np.zeros((N1, len(time)))
    c2_rec_pop = np.zeros((N2, len(time)))

    for i, t in enumerate(time):
        u_sp[:N1] = utils.spike_train(N1, u1, cbsp.TIMESTEP)
        u_sp[N1:] = utils.spike_train(N2, u2, cbsp.TIMESTEP)
        n = utils.standard_normal(N1+N2, cbsp.TIMESTEP)
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

        w1_rec_pop[:, i] = w[:N1]
        u1_rec_pop[:, i] = u_sp[:N1]
        w2_rec_pop[:, i] = w[N1:]
        u2_rec_pop[:, i] = u_sp[N1:]
        v_rec[i] = v_sp
        I_rec[i] = I
        c1_rec_pop[:, i] = c[:N1]
        c2_rec_pop[:, i] = c[N1:]

    return (w1_rec_pop, w2_rec_pop), time, (u1_rec_pop, u2_rec_pop, v_rec, I_rec, c1_rec_pop, c2_rec_pop)

# ---------------- RBP

def stdp2rbp_linear_calcium_mat(*args, **kwargs):
    """Rate-based plasticity from STDP using the linear calcium and MAT model.
    
    Args:
        - same as cbsp.population_3.linear_calcium_mat(u1, w1, u2, w2, seed)

    Returns:
        tuple: (w, v)
            with
                float: w, the population average change of synapse strength at time point 0
                float: v, the postsynaptic firing rate within the first 500ms.
    """
    (w_rec, _), t, (_, _, v_rec, _, _, _) = linear_calcium_mat(*args, **kwargs)
    return utils.derivative(w_rec.mean(axis=0), t), v_rec[0:int(0.5 / cbsp.TIMESTEP)].sum() * 2


def stdp2rbp_linear_calcium_aeif(*args, **kwargs):
    """Rate-based plasticity from STDP using the non linear calcium and MAT model.
    
    Args:
        - same as cbsp.population_3.non_linear_calcium_mat(u1, w1, u2, w2, seed)

    Returns:
        tuple: (w, v)
            with
                float: w, the population average change of synapse strength at time point 0
                float: v, the postsynaptic firing rate within the first 500ms.
    """
    (w_rec, _), t, (_, _, v_rec, _, _, _) = non_linear_calcium_mat(*args, **kwargs)
    return utils.derivative(w_rec.mean(axis=0), t), v_rec[0:int(0.5 / cbsp.TIMESTEP)].sum() * 2


def stdp2rbp_non_linear_calcium_mat(*args, **kwargs):
    """Rate-based plasticity from STDP using the linear calcium and AEIF model.
    
    Args:
        - same as cbsp.population_3.non_linear_calcium_mat(u1, w1, u2, w2, seed)

    Returns:
        tuple: (w, v)
            with
                float: w, the population average change of synapse strength at time point 0
                float: v, the postsynaptic firing rate within the first 500ms.
    """
    (w_rec, _), t, (_, _, v_rec, _, _, _) = linear_calcium_aeif(*args, **kwargs)
    return utils.derivative(w_rec.mean(axis=0), t), v_rec[0:int(0.5 / cbsp.TIMESTEP)].sum() * 2


def main_linear_calcium_mat(u1=np.arange(0, 101),
                            w1=np.arange(0, 1.05, 0.05),
                            u2=np.arange(0, 101),
                            w2=np.arange(0, 1.05, 0.05),
                            seed = np.arange(0, 100, 1),
                            nproc=2):
    """RBP from STDP for the whole parameter space using the linear calcium and MAT model.
    
    Args:
        u1 (array, optional): presynaptic firing rates for the first population. Defaults to np.arange(0, 101).
        w1 (array, optional): initial synaptic strengths for the first population. Defaults to np.arange(0, 1.05, 0.05).
        u2 (array, optional): presynaptic firing rates for the second population. Defaults to np.arange(0, 101).
        w2 (array, optional): initial synaptic strengths for the second population. Defaults to np.arange(0, 1.05, 0.05).
        seed (array, optional): random states. Defaults to np.arange(0, 100).
        nproc (int, optional): number of processes to use. Defaults to 8.
    
    Returns:
        array: rate-based plasticity and postsynaptic firing for all possible combinations of u, and w.
                Has shape (#random_states, u1.size*w1.size*u2.size*w2.size, 2).
                The last dimension is RBP and postsynaptic firing rates.
    """
    pool = multiprocessing.Pool(processes=nproc)
    results = np.zeros([seed.size, u1.size*w1.size*u2.size*w2.size, 2])
    for i, s in enumerate(tqdm(seed, desc='Seed')):
        results[i] = np.array(pool.starmap(stdp2rbp_linear_calcium_mat, product(u1, w1, u2, w2, np.array([s]))))
    return results


def main_non_linear_calcium_mat(u1=np.arange(0, 101),
                            w1=np.arange(0, 1.05, 0.05),
                            u2=np.arange(0, 101),
                            w2=np.arange(0, 1.05, 0.05),
                            seed = np.arange(0, 100, 1),
                            nproc=2):
    """
    Same as ~cbsp.population_3.main_linear_calcium_mat(u1, w1, u2, w2, seed, nproc) for the non linear calcium model.
    """
    pool = multiprocessing.Pool(processes=nproc)
    results = np.zeros([seed.size, u1.size*w1.size*u2.size*w2.size, 2])
    for i, s in enumerate(tqdm(seed, desc='Seed')):
        results[i] = np.array(pool.starmap(stdp2rbp_non_linear_calcium_mat, product(u1, w1, u2, w2, np.array([s]))))
    return results


def main_linear_calcium_aeif(u1=np.arange(0, 101),
                            w1=np.arange(0, 1.05, 0.05),
                            u2=np.arange(0, 101),
                            w2=np.arange(0, 1.05, 0.05),
                            seed = np.arange(0, 100, 1),
                            nproc=2):
    """
    Same as ~cbsp.population_3.main_linear_calcium_mat(u1, w1, u2, w2, seed, nproc) for the aeif model.
    """
    pool = multiprocessing.Pool(processes=nproc)
    results = np.zeros([seed.size, u1.size*w1.size*u2.size*w2.size, 2])
    for i, s in enumerate(tqdm(seed, desc='Seed')):
        results[i] = np.array(pool.starmap(stdp2rbp_linear_calcium_aeif, product(u1, w1, u2, w2, np.array([s]))))
    return results


def plot_3d(u1, w1, u2, w2, at_u2, at_w2, rbp_or_v, zlabel='\dot{w}'):
    """Plots the RBP or postsynaptic firing rate in 3d.
    
    Args:
        u1 (array): presynaptic firing rates of the first population.
        w1 (array): initial synapse strengths of the first population.
        u2 (array): presynaptic firing rates of the second population.
        w2 (array): initial synapse strengths of the second population.
        at_u2 (float): value in u2 for crosssection.
        at_w2 (float): value in w2 for crosssection.
        rbp_or_v (array): rate-based plasticity or postsynaptic firing rate for the parameter space.
                     Flat or of shape (u1.size, w1.size, u2.size, w2.size).
        zlabel (str): zlabel function, '${zlabel}_(u_1, w_1, u_2={at_u2}, w_2={at_w2})$'. Default is '\dot{w}'
    
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
    U, W = np.meshgrid(u1, w1)
    Z = rbp_or_v.reshape(len(u1), len(w1), len(u2), len(w2))[:, :, np.where(u2==at_u2)[0], np.where(w2==at_w2)[0]].squeeze()

    ax = plt.subplot(projection='3d')
    norm = MidpointNormalize(vmin=rbp_or_v.min()-1e-10, vmax=rbp_or_v.max(), midpoint=0)
    ax.plot_surface(X=U, Y=W, Z=Z.T, rstride=1, cstride=1, cmap=plt.cm.seismic, norm=norm, antialiased=False)
    ax.set_xlabel('$u_1$', fontsize = 10)
    ax.set_ylabel('$w_1$', fontsize = 10)
    zlabel = f'${zlabel}_(u_1, w_1, u_2={at_u2}, w_2={at_w2})$'
    ax.set_zlabel(zlabel, fontsize=10)
    ax.grid(False)
    ax.view_init(11, 130)
    return fig, ax


def plot_dynamics(w1, w2, t, u1, u2, v, I, c1, c2):
    """Plots the simulated dynamics of two populations.

    Args:
        w1 (array): change of synapse strengths in the first population. Shape (#synapses, #timesteps).
        w2 (array): change of synapse strengths in the second population. Shape (#synapses, #timesteps).
        t (array): time.
        u1 (array): presynaptic spike trains in the first population. Shape (#synapses, #timesteps).
        u2 (array): presynaptic spike trains in the second population. Shape (#synapses, #timesteps).
        v (array): postsynaptic spike train. Shape (#timesteps).
        I (array): postsynaptic current. Shape (#timesteps).
        c1 (array): calcium traces in the first population. Shape (#synapses, #timesteps).
        c2 (array): calcium traces in the second population. Shape (#synapses, #timesteps).

    Returns:
        tuple: fig, axes
    """

    fig = plt.figure(figsize=[4, 12])


    ax1 = plt.subplot(411)
    ax1.plot(t, utils.moving_average(t, u1.mean(axis=0), width=0.1, spikes=True), label='u1')
    ax1.plot(t, utils.moving_average(t, u2.mean(axis=0), width=0.1, spikes=True), label='u2')
    ax1.plot(t, utils.moving_average(t, v, width=0.5, spikes=True), label='v')
    ax1.set_xticks([])
    ax1.set_ylabel('u, v in Hz')
    ax1.set_ylim(0, 100)
    ax1.set_xlim(-0.25, 5.25)
    ax1.legend()

    ax2 = plt.subplot(412)
    ax2.plot(t, utils.moving_average(t, I, width=0.025))
    ax2.set_xticks([])
    ax2.set_ylim(0, 1.1e-9)
    ax2.set_ylabel('I in A')

    ax3 = plt.subplot(413)
    utils.trace_pop_mean_with_std(t, c1, label='population 1', fig=fig, ax=ax3)
    utils.trace_pop_mean_with_std(t, c2, label='population 2', fig=fig, ax=ax3)
    ax3.set_xticks([])
    ax3.set_ylim(0, 10)
    ax3.legend()
    ax3.set_ylabel('calcium concentration')


    ax4 = plt.subplot(414)
    utils.trace_pop_mean_with_std(t, w1, label='population 1', fig=fig, ax=ax4)
    utils.trace_pop_mean_with_std(t, w2, label='population 2', fig=fig, ax=ax4)
    ax4.set_xlabel('time')
    ax4.set_ylim(0, 1)
    ax4.legend()
    ax4.set_ylabel('synaptic strength')
    return fig, (ax1, ax2, ax3, ax4)