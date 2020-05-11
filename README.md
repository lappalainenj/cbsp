**A Theoretical Framework to Derive Simple, Firing-Rate-Dependent Mathematical Models of Synaptic Plasticity**<br>
Janne Lappalainen, Juliane Herpich, Christian Tetzlaff

Abstract: *Synaptic plasticity serves as an essential mechanism underlying cognitive processes as learning and memory. For a better understanding detailed theoretical models combine experimental underpinnings of synaptic plasticity and match experimental results. However, these models are mathematically complex impeding the comprehensive investigation of their link to cognitive processes generally executed on the neuronal network level. Here, we derive a mathematical framework enabling the simplification of such detailed models of synaptic plasticity facilitating further mathematical analyses. By this framework we obtain a compact, firing-rate-dependent mathematical formulation, which includes the essential dynamics of the detailed model and, thus, of experimentally verified properties of synaptic plasticity. Amongst others, by testing our framework by abstracting the dynamics of two well-established calcium-dependent synaptic plasticity models, we derived that the synaptic changes depend on the square of the presynaptic firing rate, which is in contrast to previous assumptions. Thus, the here-presented framework enables the derivation of biologically plausible but simple mathematical models of synaptic plasticity allowing to analyze the underlying dependencies of synaptic dynamics from neuronal properties such as the firing rate and to investigate their implications in complex neuronal networks.*

Paper: https://www.frontiersin.org/articles/10.3389/fncom.2019.00026/full

This code implements the simulation of calcium-based, spike-timing-dependent synaptic plasticity (STDP).
It contains simulations for three neural populations. They can underlie different calcium-based synaptic plasticity and neuronal dynamics.
The data resulting from STDP simulation can be fitted by weighted linear least squares regression to simplified rate-based synaptic plasticity (RBP) models. Exhaustive search can be applied to determine particularly descriptive rate-based features of calcium-based synaptic plasticity.

#### Examples

The figures presented in the paper can be reproduced step-by-step using the following notebooks:

- examples/figure_01.ipynb
- examples/figure_03.ipynb
- examples/figure_04.ipynb

Simple usage examples:
```python
# simulates stdp in population 2 with presynaptic firing of 60 Hz
# and initial synaptic strength of 0.6 a.u
# the calcium concentration follows linear dynamics [1] and the 
# postsynaptic membrane potential the adaptive-exponential integrate-and-fire model [2]
cbsp.population_2.linear_calcium_aeif(u=60, w0=0.6, seed=1)

# applies crossvalidated exhaustive search to determine the best
# three feature rule describing rate-based synaptic plasticity
es=cbsp.validation.ExhaustiveSearch(num_features=3)
es.fit(X, rbp, weights)
```

#### Installation
As operating system, Linux or macOS is recommended.
The package and its dependencies are installed in a few steps from the command line:

```
git clone https://github.com/jkoal/cbsp.git  # downloads the repository
cd cbsp  # navigates into the package directory
conda create --name cbsp python=3.7.3  # creates a virtual python environment named cbsp
conda activate cbsp  # activates the virtual environment
conda install -c anaconda pip  # just make sure pip is installed in the environment 
                               # `which pip` should point to the environment folder
pip install -e .  # installs the package and dependencies in editable mode
```

Note, this assumes conda is installed. Alternatively, pip can be used to manage virutal environments.

#### Citation
```
@article{lappalainen2019theoretical,
  title={A theoretical framework to derive simple, firing-rate-dependent mathematical models of synaptic plasticity},
  author={Lappalainen, Janne and Herpich, Juliane and Tetzlaff, Christian},
  journal={Frontiers in computational neuroscience},
  volume={13},
  pages={26},
  year={2019},
  publisher={Frontiers}
}
```

#### References

<a id="1">[1]</a>  Graupner, M., Wallisch, P., & Ostojic, S. (2016). Natural firing patterns imply low sensitivity of synaptic plasticity to spike timing compared with firing rate. Journal of Neuroscience, 36(44), 11238-11258. <br>
<a id="2">[2]</a>  Gerstner, Wulfram, and Romain Brette. "Adaptive exponential integrate-and-fire model." Scholarpedia 4.6 (2009): 8427.
