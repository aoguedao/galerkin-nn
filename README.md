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


### Domain Decomposition

- `DDQuadrature`, a child class of `Quadrature`, handles boundary ownership of boundary nodes to neighbor quadratures.
- `DDPDE`, a child class of `PDE`, takes the base PDE formulation to form the domain decomposition formulation. A Robin boundary condition is imposed into the interfaces between neighbors and it takes the solution of its neighbors to inject them as traces for the formulation. 


## Examples

[Poisson 1D: String Displacement](https://colab.research.google.com/drive/1RZQPWLb59serII9sBhm_wc35ykY8CloF?usp=sharing)


All examples are available on the folder `.\examples\`.


## Installation

For local development, clone this repository and run `python install -e .`.

For Google Colab users, they can install it in their session with 

`!pip install git+https://github.com/aoguedao/galerkin-nn`