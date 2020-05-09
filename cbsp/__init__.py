"""
Simulations of spike-timing-dependent and rate-dependent synaptic plasticity.

Lappalainen, J., Herpich, J., & Tetzlaff, C. (2019). A theoretical 
framework to derive simple, firing-rate-dependent mathematical models 
of synaptic plasticity. Frontiers in computational neuroscience, 13, 26.
"""
from cbsp import population_1
from cbsp import population_2
from cbsp import population_3
from cbsp import utils
from cbsp import validation
import pathlib
import zipfile


SIMULATION_TIME = 2.0
TIMESTEP = 0.0005


def set_simulation_time(simulation_time):
    """Updates the time for simulating STDP.

    We recompile the jitted simulation functions, 
    for updating the simulation time in the compiled code. 
    Changing SIMULATION_TIME alone has no effect after first compilation. 

    Args:
        simulation_time (float): Time in seconds to simulate STDP.
                                 Default is 2 seconds.
    """
    global SIMULATION_TIME
    SIMULATION_TIME = simulation_time
    population_1.linear_calcium.recompile()
    population_1.non_linear_calcium.recompile()
    population_2.linear_calcium_mat.recompile()
    population_2.linear_calcium_aeif.recompile()
    population_2.non_linear_calcium_mat.recompile()
    population_3.linear_calcium_mat.recompile()
    population_3.linear_calcium_aeif.recompile()
    population_3.non_linear_calcium_mat.recompile()


def set_timestep(timestep):
    """Updates the integration time steps for simulating STDP.

    We recompile the jitted simulation functions, 
    for updating the integration time steps in the compiled code. 
    Changing TIMESTEP alone has no effect after first compilation.
    
    Args:
        timestep (float): Timestep in seconds for the Euler integration.
                          Default is 0.0005 seconds.
    """
    global TIMESTEP
    TIMESTEP = timestep  
    population_1.linear_calcium.recompile()
    population_1.non_linear_calcium.recompile()
    population_2.linear_calcium_mat.recompile()
    population_2.linear_calcium_aeif.recompile()
    population_2.non_linear_calcium_mat.recompile()
    population_3.linear_calcium_mat.recompile()
    population_3.linear_calcium_aeif.recompile()
    population_3.non_linear_calcium_mat.recompile()


data_dir = pathlib.Path(__file__).parent.absolute().parent / 'data'

_files = ['p1_linear_calcium.npz',
        'p1_non_linear_calcium.npz',
        'p2_linear_calcium_aeif.npz',
        'p2_linear_calcium_mat.npz',
        'p2_non_linear_calcium_mat.npz',
        'p3_linear_calcium_aeif.npz',
        'p3_linear_calcium_mat.npz',
        'p3_non_linear_calcium_mat.npz']

_files_on_disk = sorted([f.name for f in data_dir.iterdir()])
if all([f in _files_on_disk for f in _files]):
    pass
else:
    with zipfile.ZipFile(data_dir / 'data.zip', 'r') as zip:
        zip.extractall(data_dir.parent)
