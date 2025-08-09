# Harmonic balance-automatic differentiation (HB-AD) method

This repository contains the python codes of the paper
> Yi Chen, Yuhong Jin, and Rongzhou Lin, et al. (2025). Harmonic balance-automatic differentiation method: an out-of-the-box and efficient solver for general nonlinear dynamics simulation.

This repository provides a easy-to-use, general-purpose, high efficiency platform for high-fidelity dynamic characterization of high-dimensional engineering systems based on the harmonic balance method.

## Essential Python Libraries

Following packages are required to be installed to run the above codes:

+ [PyTorch (>=2.0)](https://pytorch.org/)
+ [NumPy](https://numpy.org/)
+ [SciPy](https://www.scipy.org/)

The above package is very standard.

## Flowchart of the HB-AD method

![Flowchart of the HB-AD method](./framework.png)

## Provided examples

![Provided examples](./examples.png)

## Run a custom example

If you want to run a custom example, please rewrite the class about the system (refer to the class 'RotorSFD' in 'example_rotor_SFD.py' or the the class 'AeroEngineSystem' in 'example_aero_engine.py' ). The rest of the code requires almost no modification and does not require any manual derivation of partial derivatives.
