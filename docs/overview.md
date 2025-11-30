# Galerkin Neural Networks with Domain Decomposition

This project implements **Galerkin Neural Networks (GNNs)** for PDEs in JAX, with a focus on **domain decomposition** (overlapping Schwarz / ASM–style methods) and **problem-specific quadratures**. The goal is to combine:

- the variational structure and error-control mindset of classical Galerkin / FEM,
- the flexibility of neural networks as basis functions,
- and the scalability of domain decomposition methods.

The code is organized as a small library (`galerkinnn/`) plus a collection of self-contained examples in `examples/` that act as tests and templates.

## Core Ideas

### Galerkin Neural Networks

Instead of piecewise polynomials on a mesh, Galerkin Neural Networks use **neural networks as basis functions** in a Galerkin variational formulation. The main ingredients are:

- A **weak form** of the PDE (bilinear and linear forms).
- A finite-dimensional trial/test space spanned by neural basis functions.
- Quadrature rules to approximate the integrals in the bilinear/linear forms.
- A linear (or nonlinear) system for the Galerkin coefficients that is solved with standard numerical linear algebra.

Neural networks are treated as **parametrized basis functions**; their parameters are trained to improve the approximation quality with respect to an energy norm or residual-based objective.

### Domain Decomposition

For larger or geometrically complex problems, the domain is split into **subdomains** and solved with Schwarz/ASM-type iterations:

- The computational domain is decomposed (e.g., overlapping intervals, overlapping rectangles, disk–annulus, disk–rectangle).
- Each subdomain has its own quadrature, PDE object, and GalerkinNN problem.
- Artificial interface conditions (usually Robin/impedance-type) couple neighboring subdomains.
- A **partition of unity (PoU)** blends subdomain solutions into a global approximation.

This enables:
- localized training of neural basis functions per subdomain,
- reuse of the same machinery on each piece,
- and experiments with overlapping Schwarz iterations using neural Galerkin solvers as local solvers.

## Main Abstractions

At a high level, the project revolves around the following core types:

- `Quadrature`  
  Encodes all integration points and weights for a given domain, plus boundary points and normals. This is the main “geometry + integration” container.

- `DDQuadrature`  
  Extends `Quadrature` with metadata for domain decomposition: subdomain IDs, neighbor IDs, ownership information at the interface, and masks for global vs. artificial boundaries.

- `FunctionState`  
  Holds the **values and gradients** of the current basis functions (or solutions) evaluated at quadrature points. It decouples “evaluate the network” from “assemble bilinear forms”.

- `PDE` / `DDPDE` (formulations)  
  Dataclasses representing a specific PDE, including coefficients, forcing terms, boundary conditions, and routines to build bilinear and linear forms given a `Quadrature` or `DDQuadrature`.

- `GalerkinNN`  
  The main driver that:
  - builds and stores neural basis functions,
  - evaluates them through `FunctionState`,
  - assembles the stiffness and mass matrices,
  - solves for coefficients,
  - and exposes convenience functions to evaluate the solution.

## Typical Workflow

A typical example script under `examples/` follows this pattern:

1. **Construct quadratures**  
   - For single-domain problems: call a `*_quadrature` function (interval, rectangle, disk, etc.).
   - For DD problems: call a `dd_*_quadratures` constructor that returns a list of `DDQuadrature` objects (one per subdomain).

2. **Define the PDE**  
   - Instantiate a `PDE` or `DDPDE` dataclass with source, linear operator, bilinear form and energy norm.

3. **Create and train the GalerkinNN solver**  
   - Specify a neural network architecture, number of basis functions, and optimization parameters.
   - Use the provided training loop or a custom loop to:
     - optimize neural parameters,
     - assemble the Galerkin system,
     - and solve for coefficients.

4. **Post-processing**  
   - Evaluate the solution on a fine grid for plotting.
   - For DD problems, apply PoU blending to stitch subdomain solutions.
   - Optionally compare with an analytical solution (when available) and compute norms.

## Scope and Non-Goals

The library is intentionally focused on:

- elliptic and steady-state problems first (Poisson / steady heat),
- simple geometries (interval, rectangle, disk, annulus, disk–rectangle),
- and overlapping domain decomposition.

It is **not** intended to be a full FEM replacement or a general scientific computing framework. Meshing, adaptive refinement, and large-scale solvers are deliberately kept minimal so that the emphasis stays on:

- Galerkin formulations with neural basis functions,
- domain-decomposition algorithms,
- and clean JAX implementations that are easy to extend.
