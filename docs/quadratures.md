# Quadrature and Domain Decomposition Metadata

This document describes the `Quadrature` and `DDQuadrature` abstractions and the available quadrature constructors.

The goal is to have a **uniform interface** for all geometries and subdomains so that PDE formulations and GalerkinNN solvers can be written in a geometry-agnostic way.

## `Quadrature` Dataclass

A `Quadrature` represents all information required to approximate integrals over a domain and its boundary. The typical fields are:

- `dim: int`  
  Spatial dimension (`1` for intervals, `2` for rectangles/disk, etc.).

- `interior_x: jax.Array` with shape `(N_interior, dim)`  
  Interior quadrature points in physical coordinates.

- `interior_w: jax.Array` with shape `(N_interior,)`  
  Corresponding weights for volume integrals.

- `boundary_x: jax.Array` with shape `(N_boundary, dim)`  
  Boundary quadrature points.

- `boundary_w: jax.Array` with shape `(N_boundary,)`  
  Weights for boundary integrals (e.g. edge/arc integrals).

- `boundary_normal: jax.Array` with shape `(N_boundary, dim)`  
  Outward unit normals at boundary points.

- `boundary_tangent: jax.Array` with shape `(N_boundary, dim)`  
  Encodes which physical boundary a point belongs to (e.g. left/right/top/bottom edges, outer/inner radius, interface, etc.) so PDE formulations can apply different boundary conditions.

- `meta: dict`  
  Freeform metadata with geometry-specific information (radii, bounding boxes, overlap size, etc.) that may be useful in examples and diagnostics.

### Available Base Quadratures

The following families of quadratures are provided (names may vary slightly depending on the actual implementation):

- **1D Interval Quadrature**
  - Gauss–Legendre or Gauss–Lobatto nodes on `[a, b]`.
  - Boundary points at `x = a` and `x = b`, each with a normal pointing outward (`n = -1` on the left, `n = +1` on the right).
  - Used for 1D Poisson and steady heat problems.

- **2D Rectangle Quadrature**
  - Tensor product Gauss rules on `[a_x, b_x] × [a_y, b_y]`.
  - Boundary points split by side (left/right/bottom/top) with consistent outward normals.
  - Used for 2D Poisson/steady heat on rectangles and DD rectangle examples.

- **2D Disk Quadrature**
  - Points generated in polar coordinates `(r, θ)` and mapped to Cartesian `(x, y)`.
  - Volume weights incorporate the Jacobian factor `r`.
  - Boundary points lie on the circle of given radius with outward normals aligned with `(x, y)/‖(x, y)‖`.
  - Used for disk Poisson/steady heat benchmarks and disk components in DD problems.

- **2D Annulus / Disk Piecewise Quadrature**
  - Similar polar construction, but restricted to an annulus between inner radius `r_inner` and outer radius `r_outer`.
  - Boundary tags differentiate inner and outer circles.
  - Useful for piecewise-constant κ problems (e.g. disk with a different κ in the core vs. the annulus).

The exact function names live in `galerkinnn.quadratures` and are designed to be explicit (e.g. `interval_quadrature`, `rectangle_quadrature`, `disk_quadrature`, `disk_annulus_quadrature`, etc.).

## `DDQuadrature` Dataclass

`DDQuadrature` extends `Quadrature` with fields required for **overlapping domain decomposition**:

- `subdomain_id: int`  
  Integer index of this subdomain within a DD configuration.

- `neighbor_ids: Tuple[int, ...]`  
  Sorted tuple of neighbor subdomain IDs. The order defines the column order for one-hot encodings.

- `owner_j_at_bndry: jax.Array` with shape `(N_boundary,)`, dtype `int32`  
  For each boundary point:
  - `-1` if the point lies on a **global** boundary (physical boundary of the full domain).
  - Otherwise, the integer ID of the **owning neighbor** subdomain at that interface point.
  This is used to decide who “owns” interface data and how to apply interface conditions.

- `boundary_owner_onehot: jax.Array` with shape `(N_boundary, Jn)`  
  One-hot representation of `owner_j_at_bndry` over the `neighbor_ids`. This simplifies vectorized assembly of interface terms.

- `boundary_mask_global: jax.Array` with shape `(N_boundary,)`, dtype `bool`  
  Boolean mask selecting boundary points that lie on the global boundary versus artificial interfaces between subdomains.

All regular `Quadrature` fields (interior/boundary points, weights, normals, tags, meta) are still present and used in the same way. `DDQuadrature` just adds enough structure to:

- identify which neighbor a boundary point is associated with,
- distinguish global vs. artificial boundaries,
- and support vectorized assembly of interface Robin/impedance conditions.

## Domain Decomposition Quadrature Constructors

The library provides factory functions that construct **pairs or tuples of DDQuadratures** for standard overlapping decompositions:

- **Overlapping 1D intervals**  
  `dd_overlapping_interval_quadratures(a, b, overlap, ...)`  
  Returns two or more `DDQuadrature` objects that cover `[a, b]` with a specified overlap.

- **Overlapping rectangles**  
  `dd_overlapping_rectangle_quadratures(...)`  
  Similar idea, but for rectangular subdomains overlapping in one or both spatial directions.

- **Disk–Annulus / Disk–Disk Decompositions**  
  Functions like `dd_disk_annulus_quadratures(...)` or `dd_overlapping_disk_annulus_quadratures(...)` return a list of `DDQuadrature` objects for concentric or overlapping disk/annulus decompositions.

- **Disk–Rectangle Decompositions**  
  Constructors for configurations where a disk overlaps a rectangle. These ensure that interface boundary points are consistently parameterized on both geometries and tagged so that PoU and interface conditions can be applied.

Each constructor is responsible for:

1. Building `Quadrature` data for each subdomain.
2. Tagging global vs. interface boundaries.
3. Computing ownership arrays (`owner_j_at_bndry`, `boundary_owner_onehot`, `boundary_mask_global`).
4. Filling `meta` with any extra geometry information (overlap width, radii, bounding boxes, etc.).

This keeps PDE formulations and Schwarz iterations independent of the underlying geometry: they just iterate over a list of `DDQuadrature` objects with a consistent API.
