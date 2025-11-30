# Developer Notes

This document explains how to extend the library in three ways:

1. Add a new quadrature.
2. Add a new PDE formulation.
3. Add a new PDE example script.

The goal is to keep contributions consistent and easy to maintain.

---

## 1. Adding a New Quadrature

Location: `src/galerkinnn/quadratures/` or `galerkinnn/quadratures.py` depending on your layout.

A quadrature constructor should return a `Quadrature` object containing:

- interior_x : (N_interior, dim)
- interior_w : (N_interior,)
- boundary_x : (N_boundary, dim)
- boundary_w : (N_boundary,)
- boundary_normal : (N_boundary, dim)
- boundary_tangent : (N_boundary, dim)  # actual tangent vectors (not a tag)
- meta : dict

### Steps

(1) Define the geometry and its parameters.  
Examples: interval [a,b], rectangle, disk, annulus, etc.

(2) Build interior quadrature points.  
Use Gauss–Legendre, Gauss–Lobatto, tensor products, or polar coordinates.  
Always apply the correct Jacobian when transforming coordinates.

(3) Build boundary quadrature points.  
Split by edges or arcs. Provide:
- outward normals
- boundary tags (integers)
- weights for the boundary integral

(4) Instantiate a `Quadrature` object.  
Store any helpful information in `meta`.

(5) Add plotting / sanity checks in `examples/test_quadratures.py`.

### Domain-Decomposition Quadratures

To extend domain decomposition:

- Create a function such as `dd_overlapping_<geometry>_quadratures`.
- Internally construct base `Quadrature` objects.
- Add the DD metadata required:
  - subdomain_id
  - neighbor_ids
  - owner_j_at_bndry
  - boundary_owner_onehot
  - boundary_mask_global

Look at existing DD constructors (interval, rectangle, disk–annulus) for patterns to imitate.

---

## 2. Adding a New PDE Formulation

Location: most examples define their PDE dataclasses inline in each script for clarity. Shared/ reusable formulations can live in `src/galerkinnn/formulations.py`.

A PDE formulation is typically a dataclass that stores:
- forcing source
- boundary data
- assembly routines:
  * linear operator
  * bilinear form
  * energy norm

Assembly functions should be implemented to handle `FunctionState` instances with several states. Most examples use Einstin notation sum to speed-up the computations.

### Steps

(1) Write the weak form of the PDE.  
Example: Poisson with Robin boundary conditions.

(2) Create a PDE dataclass.  
Example:

    @struct.dataclass
    class MyPDE:
        kappa_fn: Callable
        f_fn: Callable
        alpha: float

(3) Implement assembly functions for:
- stiffness matrix contributions
- load vector contributions
- boundary contributions
- interface contributions in DD mode (Robin/impedance conditions)

These functions receive:
- a Quadrature or DDQuadrature
- a FunctionState (with four fields: interior, boundary, grad_interior, grad_boundary)

They return arrays needed by the GalerkinNN system assembly.

(4) Ensure JAX compatibility:
- use jnp, vmap, and pure functions
- avoid Python loops

(5) Optionally create a DDPDE wrapper for multi-subdomain problems.

---

## 3. Adding a New PDE Example

Location: `examples/`.

Example scripts serve as:
- documentation
- regression tests
- templates for new problems

### Steps

(1) Choose a benchmark PDE.
Specify: domain, geometry, coefficients, BCs, analytical solution when possible.

(2) Build quadrature(s).  
Call either a single-domain constructor or a DD constructor.

(3) Instantiate the PDE class.  
Pass coefficient functions and boundary parameters.

(4) Define the neural architecture.  
Set:
- widths
- activations 
- number of basis functions

(5) Instantiate a `GalerkinN` instance and then use the method `solve`.  

(6) For DD problems, implement a Schwarz iteration.  
Typical pattern:
- for each iteration: solve each subdomain, update interface data
- blend with partition of unity

(7) Post-processing.  
Evaluate solution on a fine grid and plot.  
Compare to analytical solutions if available.

---

Keeping this structure consistent across quadratures, formulations, and examples makes the library easier to maintain and contributes to reproducible numerical experiments.
