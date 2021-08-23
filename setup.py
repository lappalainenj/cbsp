from setuptools import setup, find_packages

setup(
    name="cbsp",
    version="1.0",
    packages=find_packages(),
    install_requires=["numba==0.48.0",
                        "tqdm==4.44.1",
                        "scipy==1.4.1",
                        "matplotlib==3.1.1",
                        "numpy==1.17.2",
                        "pandas==0.25.3",
                        "notebook==6.4.1",
                        "ipywidgets==7.5.1"],
    author="Janne Lappalainen",
    description="Synaptic plasticity simulations and analysis - Lappalainen J, Herpich J and Tetzlaff C (2019) doi: 10.3389/fncom.2019.00026",
)