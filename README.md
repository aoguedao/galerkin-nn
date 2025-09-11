# Galerkin Neural Networks

Object-Oriented Implementation of Galerkin Neural Networks (Ainsworth & Dong, 2021)

Original repository: https://github.com/jdongg/galerkinNNs

## Details

The main class, `GalerkinNN` takes as input a `Quadrature` and a `PDE` dataclasses as inputs.

User are responsibles for defining its own formulation based of the abstract class `PDE` with vectorized sources, linear operator, bilinear form and energy norm implementations.

The solver takes as inputs:

- A seed (for JAX reproducibility)  
- Initial guess $u_0$ which is a `FunctionState` object.
- Neural Network function which output width models $\sigma$.
- Activation function constructor
- Amount of neurons constructor
- Learning rates constructor
- Maximum amount of basis functions
- Maximum amount of epoch per basis
- Solution Tolerance
- Basis Tolerance.


## Example

[Poisson 1D: String Displacement](https://colab.research.google.com/drive/1RZQPWLb59serII9sBhm_wc35ykY8CloF?usp=sharing)

Also available on `.\examples\`.


## Installation

For local development, clone this repository and run `python install -e .`.

For Google Colab users, they can install it in their session with 

`!pip install git+https://github.com/aoguedao/galerkin-nn`