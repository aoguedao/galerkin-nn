import numpy as np
import jax
import jax.numpy as jnp

from dataclasses import replace
from flax import struct
from typing import Tuple, Dict, Any

# -------------------------
# Core Quadrature container
# -------------------------
@struct.dataclass
class Quadrature:
  dim: int
  interior_x: jax.Array                  # (N_interior, dim)
  interior_w: jax.Array                  # (N_interior,)
  boundary_x: jax.Array                  # (N_boundary, dim)
  boundary_w: jax.Array                  # (N_boundary,)
  boundary_tangent: jax.Array            # (N_boundary, dim)
  boundary_normal: jax.Array             # (N_boundary, dim)
  meta: Dict[str, Any] = struct.field(pytree_node=False)

  @property
  def n_interior(self) -> int:
    return self.interior_x.shape[0]

  @property
  def n_boundary(self) -> int:
    return self.boundary_x.shape[0]

  def integrate_interior(self, values: jax.Array) -> jax.Array:
    """
    values: (N_interior, ...) evaluated at interior_x
    returns: (...)  integral over the domain
    """
    return jnp.tensordot(self.interior_w, values, axes=1)

  def integrate_boundary(self, values: jax.Array) -> jax.Array:
    """
    values: (N_boundary, ...) evaluated at boundary_x
    returns: (...)  integral over the boundary
    """
    return jnp.tensordot(self.boundary_w, values, axes=1)


# -------------------------
# Helpers
# -------------------------
def gauss_legendre_reference(
  n: int,
  dtype=jnp.float64,
) -> tuple[jax.Array, jax.Array]:
  """
  n-point Gauss–Legendre quadrature on [-1, 1].
  Uses numpy.polynomial.legendre.leggauss under the hood (not jit'ed).
  """
  from numpy.polynomial.legendre import leggauss
  x_np, w_np = leggauss(n)
  x = jnp.asarray(x_np, dtype=dtype)
  w = jnp.asarray(w_np, dtype=dtype)
  return x, w


def map_to_interval(
  a: float,
  b: float,
  x_ref: jax.Array,
  w_ref: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """
  Map nodes/weights from [-1,1] to [a,b].
  """
  center = 0.5 * (a + b)
  half   = 0.5 * (b - a)
  x = center + half * x_ref
  w = half * w_ref
  return x, w


# -------------------------
# 1D: interval quadrature
# -------------------------
def interval_quadrature(
  bounds: Tuple[float, float],
  n_interior: int,
  dtype=jnp.float64,
) -> Quadrature:
  """
  Domain: [a, b] in 1D.
  Interior: n_interior-point Gauss–Legendre on [a,b].
  Boundary: endpoints a, b with unit weights (for ∂Ω terms).
  """
  a, b = bounds
  x_ref, w_ref = gauss_legendre_reference(n_interior, dtype=dtype)
  x_int, w_int = map_to_interval(a, b, x_ref, w_ref)

  interior_x = x_int.reshape(-1, 1)   # (n,1)
  interior_w = w_int                  # (n,)

  # boundary: just endpoints, weights = 1 for each endpoint
  boundary_x = jnp.array([[a], [b]], dtype=dtype)   # (2,1)
  boundary_w = jnp.array([1.0, 1.0], dtype=dtype)   # (2,)

  boundary_tangent = jnp.array([[1.0], [1.0]], dtype=dtype)  # arbitrary choice
  boundary_normal = jnp.array([[-1.0], [1.0]], dtype=dtype)  # outward normals

  meta = {
    "kind": "interval",
    "bounds": (a, b),
    "n_interior": n_interior,
  }

  return Quadrature(
    dim=1,
    interior_x=interior_x,
    interior_w=interior_w,
    boundary_x=boundary_x,
    boundary_w=boundary_w,
    boundary_tangent=boundary_tangent,
    boundary_normal=boundary_normal,
    meta=meta,
  )


# -------------------------
# 2D: rectangle quadrature
# -------------------------
def rectangle_quadrature(
  bounds: Tuple[Tuple[float, float], Tuple[float, float]],
  nx: int,
  ny: int,
  n_edge: int | None = None,
  dtype=jnp.float64,
) -> Quadrature:
  """
  Domain: [ax,bx] x [ay,by] in 2D.
  Interior: tensor-product GL with (nx, ny) points.
  Boundary: GL with n_edge points on each edge, scaled by ds.
  If n_edge is None, use n_edge = max(nx, ny).
  """
  (ax, bx), (ay, by) = bounds
  if n_edge is None:
    n_edge = max(nx, ny)

  # --- interior: tensor GL on rectangle ---

  x_ref, w_ref_x = gauss_legendre_reference(nx, dtype=dtype)
  y_ref, w_ref_y = gauss_legendre_reference(ny, dtype=dtype)

  x_int, w_x = map_to_interval(ax, bx, x_ref, w_ref_x)
  y_int, w_y = map_to_interval(ay, by, y_ref, w_ref_y)

  X, Y = jnp.meshgrid(x_int, y_int, indexing="ij")   # (nx, ny)
  interior_x = jnp.stack([X.ravel(), Y.ravel()], axis=1)  # (nx*ny, 2)

  # weights: tensor-product, w_x[i] w_y[j]
  interior_w = jnp.outer(w_x, w_y).ravel()          # (nx*ny,)

  # --- boundary: four edges with 1D GL on each side ---

  s_ref, w_ref_s = gauss_legendre_reference(n_edge, dtype=dtype)

  boundary_points = []
  boundary_weights = []
  tangents = []
  normals = []

  # left edge: x=ax, y∈[ay,by], outward normal = (-1,0)
  y_left, w_y_edge = map_to_interval(ay, by, s_ref, w_ref_s)   # includes dy
  x_left = jnp.full_like(y_left, ax)
  ds_left = w_y_edge                                          # dy = ds here

  boundary_points.append(jnp.stack([x_left, y_left], axis=1))
  boundary_weights.append(ds_left)
  tangents.append(jnp.stack([jnp.zeros_like(y_left), jnp.ones_like(y_left)], axis=1))
  normals.append(jnp.stack([-jnp.ones_like(y_left), jnp.zeros_like(y_left)], axis=1))

  # right edge: x=bx, y∈[ay,by], outward normal = (1,0)
  y_right, w_y_edge = map_to_interval(ay, by, s_ref, w_ref_s)
  x_right = jnp.full_like(y_right, bx)
  ds_right = w_y_edge

  boundary_points.append(jnp.stack([x_right, y_right], axis=1))
  boundary_weights.append(ds_right)
  tangents.append(jnp.stack([jnp.zeros_like(y_right), jnp.ones_like(y_right)], axis=1))
  normals.append(jnp.stack([jnp.ones_like(y_right), jnp.zeros_like(y_right)], axis=1))

  # bottom edge: y=ay, x∈[ax,bx], outward normal = (0,-1)
  x_bottom, w_x_edge = map_to_interval(ax, bx, s_ref, w_ref_s)
  y_bottom = jnp.full_like(x_bottom, ay)
  ds_bottom = w_x_edge

  boundary_points.append(jnp.stack([x_bottom, y_bottom], axis=1))
  boundary_weights.append(ds_bottom)
  tangents.append(jnp.stack([jnp.ones_like(x_bottom), jnp.zeros_like(x_bottom)], axis=1))
  normals.append(jnp.stack([jnp.zeros_like(x_bottom), -jnp.ones_like(x_bottom)], axis=1))

  # top edge: y=by, x∈[ax,bx], outward normal = (0,1)
  x_top, w_x_edge = map_to_interval(ax, bx, s_ref, w_ref_s)
  y_top = jnp.full_like(x_top, by)
  ds_top = w_x_edge

  boundary_points.append(jnp.stack([x_top, y_top], axis=1))
  boundary_weights.append(ds_top)
  tangents.append(jnp.stack([jnp.ones_like(x_top), jnp.zeros_like(x_top)], axis=1))
  normals.append(jnp.stack([jnp.zeros_like(x_top), jnp.ones_like(x_top)], axis=1))

  boundary_x = jnp.concatenate(boundary_points, axis=0)          # (4*n_edge, 2)
  boundary_w = jnp.concatenate(boundary_weights, axis=0)         # (4*n_edge,)
  boundary_tangent = jnp.concatenate(tangents, axis=0)           # (4*n_edge, 2)
  boundary_normal = jnp.concatenate(normals, axis=0)             # (4*n_edge, 2)

  meta = {
    "kind": "rectangle",
    "bounds": bounds,
    "nx": nx,
    "ny": ny,
    "n_edge": n_edge,
  }

  return Quadrature(
    dim=2,
    interior_x=interior_x,
    interior_w=interior_w,
    boundary_x=boundary_x,
    boundary_w=boundary_w,
    boundary_tangent=boundary_tangent,
    boundary_normal=boundary_normal,
    meta=meta,
  )


# -------------------------
# 2D: disk quadrature
# -------------------------
def disk_quadrature(
  radius: float,
  n_r: int,
  n_theta: int,
  dtype=jnp.float64,
) -> Quadrature:
  """
  Domain: disk of radius R in 2D, centered at the origin.
  Interior: polar coords with GL in r∈[0,R] and θ∈[0,2π]; weights include r.
  Boundary: circle r=R, GL in θ with ds = R dθ.
  """
  R = radius

  # 1D GL in r on [0,R]
  r_ref, w_ref_r = gauss_legendre_reference(n_r, dtype=dtype)
  r_nodes, w_r = map_to_interval(0.0, R, r_ref, w_ref_r)   # approximates ∫_0^R ...
  # 1D GL in theta on [0,2π]
  theta_ref, w_ref_theta = gauss_legendre_reference(n_theta, dtype=dtype)
  theta_nodes, w_theta = map_to_interval(0.0, 2.0 * jnp.pi, theta_ref, w_ref_theta)

  # interior points in polar -> Cartesian
  R_grid, Theta_grid = jnp.meshgrid(r_nodes, theta_nodes, indexing="ij")
  X = R_grid * jnp.cos(Theta_grid)
  Y = R_grid * jnp.sin(Theta_grid)
  interior_x = jnp.stack([X.ravel(), Y.ravel()], axis=1)  # (n_r*n_theta, 2)

  # weights: w_r[i] w_theta[j] * r_i (Jacobian r dr dθ)
  # R_grid = r_i, shape (n_r, n_theta)
  W = jnp.outer(w_r, w_theta) * R_grid                   # (n_r, n_theta)
  interior_w = W.ravel()

  # boundary: circle r=R, θ in [0,2π]
  # boundary points
  x_b = R * jnp.cos(theta_nodes)
  y_b = R * jnp.sin(theta_nodes)
  boundary_x = jnp.stack([x_b, y_b], axis=1)             # (n_theta, 2)

  # ds = R dθ
  boundary_w = R * w_theta                               # (n_theta,)

  # tangents: unit tangent along circle (counterclockwise)
  t_x = -jnp.sin(theta_nodes)
  t_y =  jnp.cos(theta_nodes)
  boundary_tangent = jnp.stack([t_x, t_y], axis=1)

  # normals: outward unit normal = (cosθ, sinθ)
  n_x = jnp.cos(theta_nodes)
  n_y = jnp.sin(theta_nodes)
  boundary_normal = jnp.stack([n_x, n_y], axis=1)

  meta = {
    "kind": "disk",
    "radius": radius,
    "n_r": n_r,
    "n_theta": n_theta,
  }

  return Quadrature(
    dim=2,
    interior_x=interior_x,
    interior_w=interior_w,
    boundary_x=boundary_x,
    boundary_w=boundary_w,
    boundary_tangent=boundary_tangent,
    boundary_normal=boundary_normal,
    meta=meta,
  )


# -------------------------
# 2D: annulus quadrature
# -------------------------
def annulus_quadrature(
  r_inner: float,
  r_outer: float,
  n_r: int,
  n_theta: int,
  dtype=jnp.float64,
) -> Quadrature:
  """
  Domain: annulus { (x,y): r_inner <= r <= r_outer }.
  Interior: polar coords with GL in r∈[r_inner,R_outer], θ∈[0,2π]; weights include r.
  Boundary: two circles r=r_inner (inner) and r=r_outer (outer).
           Outward normal points outward from the domain (so inner circle normal points inward in Cartesian coordinates).
  """
  Rin, Rout = r_inner, r_outer

  # 1D GL in r on [Rin, Rout]
  r_ref, w_ref_r = gauss_legendre_reference(n_r, dtype=dtype)
  r_nodes, w_r = map_to_interval(Rin, Rout, r_ref, w_ref_r)   # ∫_Rin^Rout ...

  # 1D GL in θ on [0,2π]
  theta_ref, w_ref_theta = gauss_legendre_reference(n_theta, dtype=dtype)
  theta_nodes, w_theta = map_to_interval(0.0, 2.0 * jnp.pi, theta_ref, w_ref_theta)

  # interior: polar -> Cartesian
  R_grid, Theta_grid = jnp.meshgrid(r_nodes, theta_nodes, indexing="ij")
  X = R_grid * jnp.cos(Theta_grid)
  Y = R_grid * jnp.sin(Theta_grid)
  interior_x = jnp.stack([X.ravel(), Y.ravel()], axis=1)

  # weights: w_r[i] w_theta[j] * r_i
  W = jnp.outer(w_r, w_theta) * R_grid
  interior_w = W.ravel()

  # boundary: inner and outer circles

  # θ is shared
  cos_theta = jnp.cos(theta_nodes)
  sin_theta = jnp.sin(theta_nodes)

  # outer circle: r = Rout, outward normal = (cosθ, sinθ)
  x_outer = Rout * cos_theta
  y_outer = Rout * sin_theta
  boundary_x_outer = jnp.stack([x_outer, y_outer], axis=1)
  boundary_w_outer = Rout * w_theta   # ds = Rout dθ

  t_outer_x = -sin_theta
  t_outer_y =  cos_theta
  boundary_tangent_outer = jnp.stack([t_outer_x, t_outer_y], axis=1)

  n_outer_x = cos_theta
  n_outer_y = sin_theta
  boundary_normal_outer = jnp.stack([n_outer_x, n_outer_y], axis=1)

  # inner circle: r = Rin, outward normal is *towards the hole exterior*, so points inward in xy-plane
  x_inner = Rin * cos_theta
  y_inner = Rin * sin_theta
  boundary_x_inner = jnp.stack([x_inner, y_inner], axis=1)
  boundary_w_inner = Rin * w_theta   # ds = Rin dθ

  t_inner_x = -sin_theta
  t_inner_y =  cos_theta
  boundary_tangent_inner = jnp.stack([t_inner_x, t_inner_y], axis=1)

  # For the annulus, outward from the domain is:
  # - outer circle: +(cosθ, sinθ)
  # - inner circle: -(cosθ, sinθ)
  n_inner_x = -cos_theta
  n_inner_y = -sin_theta
  boundary_normal_inner = jnp.stack([n_inner_x, n_inner_y], axis=1)

  # concatenate inner first, then outer
  boundary_x = jnp.concatenate([boundary_x_inner, boundary_x_outer], axis=0)
  boundary_w = jnp.concatenate([boundary_w_inner, boundary_w_outer], axis=0)
  boundary_tangent = jnp.concatenate(
    [boundary_tangent_inner, boundary_tangent_outer], axis=0
  )
  boundary_normal = jnp.concatenate(
    [boundary_normal_inner, boundary_normal_outer], axis=0
  )

  meta = {
    "kind": "annulus",
    "r_inner": r_inner,
    "r_outer": r_outer,
    "n_r": n_r,
    "n_theta": n_theta,
  }

  return Quadrature(
    dim=2,
    interior_x=interior_x,
    interior_w=interior_w,
    boundary_x=boundary_x,
    boundary_w=boundary_w,
    boundary_tangent=boundary_tangent,
    boundary_normal=boundary_normal,
    meta=meta,
  )


# =====================
# Domain Decomposition
# =====================
@struct.dataclass
class DDQuadrature(Quadrature):
  """
  Quadrature with domain-decomposition metadata extending `Quadrature`.

  Fields aligned with boundary_x (Nb points):
  - subdomain_id: integer id of this subdomain (static)
  - neighbor_ids: (Jn,) tuple of neighbor subdomain ids (static; order defines columns)
  - owner_j_at_bndry: (Nb,) int32; neighbor id that owns that boundary point; -1 if global boundary
  - boundary_owner_onehot: (Nb, Jn) float32; one-hot over neighbor_ids; zeros row on global boundary
  - boundary_mask_global: (Nb,) bool; True where boundary point is a global boundary
  """

  # static (non-pytree) metadata
  subdomain_id: int = struct.field(pytree_node=False)
  neighbor_ids: Tuple[int, ...] = struct.field(pytree_node=False)

  # pytree leaves (arrays)
  owner_j_at_bndry: jax.Array          # (Nb,), int32; -1 on global boundary
  boundary_owner_onehot: jax.Array     # (Nb, Jn), float32 one-hot; zeros on global boundary
  boundary_mask_global: jax.Array      # (Nb,), bool

  @classmethod
  def from_quadrature(
    cls,
    quad: Quadrature,
    *,
    subdomain_id: int,
    neighbor_ids: Tuple[int, ...],
    owner_j_at_bndry: jax.Array,
    meta_update: Dict[str, Any] | None = None,
  ) -> "DDQuadrature":
    """
    Lift a plain `Quadrature` into a `DDQuadrature` by attaching DD metadata.

    - `owner_j_at_bndry`: (Nb,) int32, neighbor id or -1 for global boundary.
    - `neighbor_ids`: tuple of neighbor ids; defines the columns of the one-hot.
    - `meta_update`: optional dict merged into quad.meta.
    """
    # DD-specific arrays
    onehot = _owner_onehot_from_ids(owner_j_at_bndry, neighbor_ids)
    mask_g = (owner_j_at_bndry == -1)

    # updated meta (copy to keep things immutable)
    if quad.meta is None:
      new_meta: Dict[str, Any] = {}
    else:
      new_meta = dict(quad.meta)
    if meta_update is not None:
      new_meta.update(meta_update)

    return cls(
      dim=quad.dim,
      interior_x=quad.interior_x,
      interior_w=quad.interior_w,
      boundary_x=quad.boundary_x,
      boundary_w=quad.boundary_w,
      boundary_tangent=quad.boundary_tangent,
      boundary_normal=quad.boundary_normal,
      meta=new_meta,
      subdomain_id=subdomain_id,
      neighbor_ids=neighbor_ids,
      owner_j_at_bndry=owner_j_at_bndry,
      boundary_owner_onehot=onehot,
      boundary_mask_global=mask_g,
    )


def _owner_onehot_from_ids(
  owner_j_at_bndry: jax.Array,
  neighbor_ids: Tuple[int, ...],
) -> jax.Array:
  """
  owner_j_at_bndry: (Nb,) int32 with neighbor ids, -1 for global boundary
  neighbor_ids: tuple length Jn
  returns (Nb, Jn) one-hot over neighbor_ids (zeros row when owner == -1)
  """
  nb = owner_j_at_bndry.shape[0]
  nid = jnp.array(neighbor_ids, dtype=jnp.int32)  # (Jn,)
  # Broadcast equality; rows with -1 yield all False -> zeros row
  oh = (owner_j_at_bndry.reshape(nb, 1) == nid.reshape(1, -1)).astype(jnp.float32)
  return oh

# -------------------------
# DD Interval: 2 Intervals
# -------------------------
def dd_overlapping_interval_quadratures(
  bounds: Tuple[float, float] = (0.0, 1.0),
  mid: float = 0.5,
  overlap: float = 0.2,
  n_interior: int = 64,
):
  """
  Build two overlapping subdomain quadratures on [a,b]:
    sub 0: [a, mid + overlap/2]
    sub 1: [mid - overlap/2, b]

  Each DDQuadrature has neighbor_ids, owner_j_at_bndry, boundary_owner_onehot, boundary_mask_global.

  Geometry responsibility (points/weights/normals) is handled by `interval_quadrature`.
  DD metadata here only labels which boundary points are global vs interface-owned.
  """
  a, b = bounds
  if not (b > a):
    raise ValueError("Invalid interval bounds.")
  if not (0.0 < overlap < (b - a)):
    raise ValueError("overlap must be in (0, b-a)")

  left_b  = float(jnp.clip(mid + 0.5 * overlap, a, b))
  right_a = float(jnp.clip(mid - 0.5 * overlap, a, b))
  if not (right_a < left_b):
    raise ValueError("Invalid overlap: subdomains do not overlap.")

  # Base quadratures (each has two boundary points: left, right)
  q0 = interval_quadrature((a, left_b), n_interior=n_interior)
  q1 = interval_quadrature((right_a, b), n_interior=n_interior)

  # --- Subdomain 0 metadata ---
  sub0_id    = 0
  sub0_neigh = (1,)  # only neighbor is subdomain 1
  # boundary_x order from interval_quadrature: [left_endpoint, right_endpoint]
  # Ownership: left boundary is global, right boundary is "owned" by subdomain 1.
  sub0_owner = jnp.array([-1, 1], dtype=jnp.int32)        # (Nb=2,)
  Q0 = DDQuadrature.from_quadrature(
    q0,
    subdomain_id=sub0_id,
    neighbor_ids=sub0_neigh,
    owner_j_at_bndry=sub0_owner,
  )

  # --- Subdomain 1 metadata ---
  sub1_id    = 1
  sub1_neigh = (0,)
  # boundary_x order: [left_endpoint, right_endpoint]
  # Ownership: left boundary owned by subdomain 0, right boundary is global.
  sub1_owner = jnp.array([0, -1], dtype=jnp.int32)
  Q1 =  DDQuadrature.from_quadrature(
    q1,
    subdomain_id=sub1_id,
    neighbor_ids=sub1_neigh,
    owner_j_at_bndry=sub1_owner
  )

  return Q0, Q1

# -------------------------
# DD Rectangle: 2 Rectangles
# -------------------------
def dd_overlapping_rectangle_quadratures(
  bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.0, 1.0), (0.0, 1.0)),
  mid: float = 0.5,
  overlap: float = 0.2,
  nx: int = 16,
  ny: int = 16,
  n_edge: int | None = None,
):
  """
  Build two overlapping rectangular subdomain quadratures on [a,b]x[c,d]:

    sub 0: [a, mid + overlap/2] x [c,d]
    sub 1: [mid - overlap/2, b] x [c,d]

  Each DDQuadrature has neighbor_ids, owner_j_at_bndry, boundary_owner_onehot, boundary_mask_global.

  Assumes `rectangle_quadrature` builds boundary points in the order:
    [left edge, right edge, bottom edge, top edge],
  each edge with `n_edge` points.
  """
  (a, b), (c, d) = bounds
  if not (b > a and d > c):
    raise ValueError("Invalid bounds for rectangle.")
  if not (0.0 < overlap < (b - a)):
    raise ValueError("overlap must be in (0, b-a)")

  left_b  = float(jnp.clip(mid + 0.5 * overlap, a, b))
  right_a = float(jnp.clip(mid - 0.5 * overlap, a, b))
  if not (right_a < left_b):
    raise ValueError("Invalid overlap: subdomains do not overlap.")

  # Build geometric quadratures
  q0 = rectangle_quadrature(((a, left_b), (c, d)), nx=nx, ny=ny, n_edge=n_edge)
  q1 = rectangle_quadrature(((right_a, b), (c, d)), nx=nx, ny=ny, n_edge=n_edge)

  # Extract edge size from meta (we stored n_edge there)
  edge0 = q0.meta["n_edge"]
  edge1 = q1.meta["n_edge"]

  nb0 = q0.boundary_x.shape[0]  # should be 4*edge0
  nb1 = q1.boundary_x.shape[0]  # should be 4*edge1

  # --- Subdomain 0 metadata ---
  sub0_id    = 0
  sub0_neigh = (1,)

  # Boundary ordering: [left(0:edge), right(edge:2edge), bottom(2edge:3edge), top(3edge:4edge)]
  # For subdomain 0: left, bottom, top global; right shared with subdomain 1.
  owner0 = jnp.full((nb0,), -1, dtype=jnp.int32)
  owner0 = owner0.at[edge0:2 * edge0].set(1)  # right edge owned by subdomain 1
  Q0 = DDQuadrature.from_quadrature(
    q0,
    subdomain_id=sub0_id,
    neighbor_ids=sub0_neigh,
    owner_j_at_bndry=owner0,
  )

  # --- Subdomain 1 metadata ---
  sub1_id    = 1
  sub1_neigh = (0,)
  # For subdomain 1: right, bottom, top global; left shared with subdomain 0.
  owner1 = jnp.full((nb1,), -1, dtype=jnp.int32)
  owner1 = owner1.at[:edge1].set(0)  # left edge owned by subdomain 0
  Q1 = DDQuadrature.from_quadrature(
    q1,
    subdomain_id=sub1_id,
    neighbor_ids=sub1_neigh,
    owner_j_at_bndry=owner1,
  )

  return Q0, Q1


# -------------------------
# DD Disk: Disk + Annulus
# -------------------------
def dd_disk_annulus_quadratures(
  R: float,
  r: float,
  n_r_disk: int,
  n_r_annulus: int,
  n_theta: int,
  dtype=jnp.float64,
) -> Tuple[DDQuadrature, DDQuadrature]:
  """
  Domain decomposition of a disk of radius R into:
    - subdomain 0: disk of radius r    (centered at origin)
    - subdomain 1: annulus r <= |x| <= R

  They share the circle |x| = r as an interface.
  Only the outer circle |x| = R is a global physical boundary.

  Both subdomains use the same n_theta angular quadrature so that
  interface points line up.
  """

  if not (0.0 < r < R):
    raise ValueError("Require 0 < r < R for disk-annulus decomposition.")

  # --- Build geometric quadratures ---

  q_disk = disk_quadrature(
    radius=r,
    n_r=n_r_disk,
    n_theta=n_theta,
    dtype=dtype,
  )

  q_ann = annulus_quadrature(
    r_inner=r,
    r_outer=R,
    n_r=n_r_annulus,
    n_theta=n_theta,
    dtype=dtype,
  )

  # q_disk.boundary_x: n_theta points on circle r
  # q_ann.boundary_x: 2*n_theta points, inner circle first, then outer circle
  nb_disk = q_disk.boundary_x.shape[0]          # = n_theta
  nb_ann  = q_ann.boundary_x.shape[0]           # = 2*n_theta

  # -------------------------------
  # Subdomain 0: inner disk (radius r)
  # -------------------------------
  sub0_id    = 0
  sub0_neigh = (1,)   # only neighbor is annulus
  # All boundary points of the disk lie on the interface with subdomain 1.
  # Use the same convention as before: owner_j_at_bndry is the neighbor
  # that "owns" that point; global boundary => -1.
  # Here there is no global boundary for the disk.
  owner0 = jnp.full((nb_disk,), 1, dtype=jnp.int32)  # all interface, owned by neighbor 1
  Q_disk = DDQuadrature.from_quadrature(
    q_disk,
    subdomain_id=sub0_id,
    neighbor_ids=sub0_neigh,
    owner_j_at_bndry=owner0,
  )

  # -------------------------------
  # Subdomain 1: annulus (r <= |x| <= R)
  # -------------------------------
  sub1_id    = 1
  sub1_neigh = (0,)   # only neighbor is inner disk

  # annulus boundary ordering from annulus_quadrature:
  # [inner circle (index 0:n_theta), outer circle (index n_theta:2*n_theta)]
  n_theta_ann = q_ann.meta["n_theta"]
  if n_theta_ann != n_theta:
    raise ValueError("disk_quadrature and annulus_quadrature must use the same n_theta.")

  owner1 = jnp.full((nb_ann,), -1, dtype=jnp.int32)  # default: global
  # inner circle (interface with disk) is owned by subdomain 0
  owner1 = owner1.at[:n_theta].set(0)
  Q_ann = DDQuadrature.from_quadrature(
    q_ann,
    subdomain_id=sub1_id,
    neighbor_ids=sub1_neigh,
    owner_j_at_bndry=owner1,
  )

  return Q_disk, Q_ann


def dd_overlapping_disk_annulus_quadratures(
  R: float,
  r: float,
  overlap: float,
  n_r_disk: int,
  n_r_annulus: int,
  n_theta: int,
  dtype=jnp.float64,
) -> Tuple[DDQuadrature, DDQuadrature]:
  """
  Domain decomposition of the disk {|x| <= R} into TWO OVERLAPPING subdomains:

    sub 0 (inner):  disk of radius r_disk = r + overlap/2
    sub 1 (outer):  annulus r_ann_in <= |x| <= R,
                    where r_ann_in = r - overlap/2.

  They overlap in the radial band [r - overlap/2, r + overlap/2].
  Only the outer circle |x| = R is a global (physical) boundary.
  All inner circles are artificial interfaces.

  Assumes disk_quadrature(...) and annulus_quadrature(...) exist and that
  annulus_quadrature orders boundary points as:
      [inner circle (n_theta), outer circle (n_theta)].
  """

  if not (R > 0.0):
    raise ValueError("R must be positive.")
  if not (0.0 < r < R):
    raise ValueError("Require 0 < r < R.")
  if not (overlap > 0.0):
    raise ValueError("overlap must be positive.")

  # Constraints: r - overlap/2 > 0, r + overlap/2 < R
  max_overlap = 2.0 * min(r, R - r)
  if not (overlap < max_overlap):
    raise ValueError(
      f"overlap too large; require overlap < {max_overlap} to keep radii inside (0,R)."
    )

  r_disk     = float(r + 0.5 * overlap)
  r_ann_in   = float(r - 0.5 * overlap)
  r_ann_out  = float(R)

  # --- Build geometric quadratures ---

  q_disk = disk_quadrature(
    radius=r_disk,
    n_r=n_r_disk,
    n_theta=n_theta,
    dtype=dtype,
  )

  q_ann = annulus_quadrature(
    r_inner=r_ann_in,
    r_outer=r_ann_out,
    n_r=n_r_annulus,
    n_theta=n_theta,
    dtype=dtype,
  )

  nb_disk = q_disk.boundary_x.shape[0]         # = n_theta
  nb_ann  = q_ann.boundary_x.shape[0]          # = 2*n_theta

  # Sanity on n_theta from meta if stored there
  if "n_theta" in q_disk.meta:
    assert q_disk.meta["n_theta"] == n_theta
  if "n_theta" in q_ann.meta:
    assert q_ann.meta["n_theta"] == n_theta

  # ---------------------------------
  # Subdomain 0: overlapping disk
  # ---------------------------------
  # All boundary points of the inner disk are artificial interface,
  # owned by subdomain 1.
  sub0_owner = jnp.full((nb_disk,), 1, dtype=jnp.int32)  # neighbor id 1
  Q_disk_dd = DDQuadrature.from_quadrature(
    q_disk,
    subdomain_id=0,
    neighbor_ids=(1,),
    owner_j_at_bndry=sub0_owner,
    meta_update={
      "dd_kind": "disk_annulus_overlap",
      "role": "inner_disk",
      "R_global": R,
      "r_mid": r,
      "overlap": overlap,
    },
  )

  # ---------------------------------
  # Subdomain 1: overlapping annulus
  # ---------------------------------
  # annulus boundary ordering: [inner circle (0:n_theta), outer circle (n_theta:2*n_theta)]
  n_theta_ann = nb_ann // 2
  if n_theta_ann != n_theta:
    raise ValueError("Annulus boundary ordering / n_theta mismatch.")

  owner1 = jnp.full((nb_ann,), -1, dtype=jnp.int32)  # default: global
  # inner circle is interface with subdomain 0
  owner1 = owner1.at[:n_theta_ann].set(0)

  Q_ann_dd = DDQuadrature.from_quadrature(
    q_ann,
    subdomain_id=1,
    neighbor_ids=(0,),
    owner_j_at_bndry=owner1,
    meta_update={
      "dd_kind": "disk_annulus_overlap",
      "role": "outer_annulus",
      "R_global": R,
      "r_mid": r,
      "overlap": overlap,
    },
  )

  return Q_disk_dd, Q_ann_dd
