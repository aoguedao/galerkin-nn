import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple
from dataclasses import dataclass
from scipy.special import roots_jacobi, eval_jacobi


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Quadrature:
  dim: int
  shape: Tuple[int, ...]                 # (ng,) for 1D, (nr, nt) for 2D disk, etc.
  interior_x: jax.Array                  # (N_interior, dim)
  interior_w: jax.Array                  # (N_interior,)
  boundary_x: jax.Array                  # (N_boundary, dim)
  boundary_w: jax.Array                  # (N_boundary,)
  boundary_tangent: jax.Array            # (N_boundary, dim)
  boundary_normal: jax.Array             # (N_boundary, dim)

  def integrate_interior(self, values: jax.Array) -> jax.Array:
    return jnp.sum(values * self.interior_w[:, None], axis=0, keepdims=True)

  def integrate_boundary(self, values: jax.Array) -> jax.Array:
    return jnp.sum(values * self.boundary_w[:, None], axis=0, keepdims=True)


def gauss_lobatto(ng: int):
  if ng < 2:
    raise ValueError("Gauss–Lobatto requires ng >= 2")
  x = np.zeros([ng,])
  w = np.zeros([ng,])
  x[0], x[-1] = -1.0, 1.0
  w[0] = w[-1] = 2.0 / (ng * (ng - 1.0))
  if ng > 2:
    xint, _ = roots_jacobi(ng - 2, 1.0, 1.0)
    x[1:-1] = np.squeeze(xint)
    w[1:-1] = np.squeeze(
      2.0 / (ng * (ng - 1.0) * eval_jacobi(ng - 1, 0.0, 0.0, xint) ** 2)
    )
  return x, w


def gauss_legendre_interval_quadrature(bounds: Tuple[float, float], ng: int) -> Quadrature:
  a, b = bounds
  x_ref, w_ref = gauss_lobatto(ng)
  x_mapped = 0.5 * (b - a) * x_ref + 0.5 * (b + a)
  w_mapped = 0.5 * (b - a) * w_ref

  interior_x = jnp.array(x_mapped).reshape(-1, 1)
  interior_w = jnp.array(w_mapped)

  boundary_x = jnp.array([[a], [b]])
  boundary_w = jnp.array([1.0, 1.0])
  boundary_tangent = jnp.array([[1.0], [1.0]])
  boundary_normal = jnp.array([[-1.0], [1.0]])

  return Quadrature(
    dim=1,
    shape=(ng,),
    interior_x=interior_x,
    interior_w=interior_w,
    boundary_x=boundary_x,
    boundary_w=boundary_w,
    boundary_tangent=boundary_tangent,
    boundary_normal=boundary_normal
  )



def gauss_lobatto_rectangle_quadrature(
  bounds: Tuple[Tuple[float, float], Tuple[float, float]], ng: int
) -> Quadrature:
  (a, b), (c, d) = bounds
  nx, ny = ng, ng
  x0, w0 = gauss_lobatto(nx)
  x1, w1 = gauss_lobatto(ny)

  x_mapped = 0.5 * (b - a) * x0 + 0.5 * (b + a)
  y_mapped = 0.5 * (d - c) * x1 + 0.5 * (d + c)
  w_x = 0.5 * (b - a) * jnp.array(w0)
  w_y = 0.5 * (d - c) * jnp.array(w1)

  X, Y = jnp.meshgrid(jnp.array(x_mapped), jnp.array(y_mapped), indexing='ij')
  interior_x = jnp.stack([X.ravel(), Y.ravel()], axis=1)
  interior_w = jnp.outer(w_x, w_y).ravel()

  boundary_points, boundary_weights, tangents, normals = [], [], [], []

  # left edge (x=a)
  y = jnp.array(y_mapped)
  boundary_points.append(jnp.stack([jnp.full_like(y, a), y], axis=1))
  boundary_weights.append(w_y)
  tangents.append(jnp.stack([jnp.zeros_like(y), jnp.ones_like(y)], axis=1))
  normals.append(jnp.stack([-jnp.ones_like(y), jnp.zeros_like(y)], axis=1))

  # right edge (x=b)
  boundary_points.append(jnp.stack([jnp.full_like(y, b), y], axis=1))
  boundary_weights.append(w_y)
  tangents.append(jnp.stack([jnp.zeros_like(y), -jnp.ones_like(y)], axis=1))
  normals.append(jnp.stack([jnp.ones_like(y), jnp.zeros_like(y)], axis=1))

  # bottom edge (y=c)
  x = jnp.array(x_mapped)
  boundary_points.append(jnp.stack([x, jnp.full_like(x, c)], axis=1))
  boundary_weights.append(w_x)
  tangents.append(jnp.stack([-jnp.ones_like(x), jnp.zeros_like(x)], axis=1))
  normals.append(jnp.stack([jnp.zeros_like(x), -jnp.ones_like(x)], axis=1))

  # top edge (y=d)
  boundary_points.append(jnp.stack([x, jnp.full_like(x, d)], axis=1))
  boundary_weights.append(w_x)
  tangents.append(jnp.stack([jnp.ones_like(x), jnp.zeros_like(x)], axis=1))
  normals.append(jnp.stack([jnp.zeros_like(x), jnp.ones_like(x)], axis=1))

  boundary_x = jnp.concatenate(boundary_points, axis=0)
  boundary_w = jnp.concatenate(boundary_weights, axis=0)
  boundary_tangent = jnp.concatenate(tangents, axis=0)
  boundary_normal = jnp.concatenate(normals, axis=0)

  return Quadrature(
    ng,
    2,
    interior_x,
    interior_w,
    boundary_x,
    boundary_w,
    boundary_tangent,
    boundary_normal
  )


def gauss_legendre_disk_quadrature(nr: int, nt: int, R: float = 1.0) -> Quadrature:
  # Radial quadrature on [0,R]
  r_nodes, r_weights = roots_jacobi(nr, 0.0, 0.0)   # Gauss–Legendre
  r = 0.5 * R * (r_nodes + 1.0)                     # map [-1,1] → [0,R]
  wr = 0.5 * R * r_weights * r                      # includes dr and Jacobian r

  # Angular quadrature on [0,2π]
  t_nodes, t_weights = roots_jacobi(nt, 0.0, 0.0)
  theta = np.pi * (t_nodes + 1.0)                   # map [-1,1] → [0,2π]
  wt = np.pi * t_weights

  # Tensor product grid
  Rgrid, Tgrid = np.meshgrid(r, theta, indexing="ij")
  WR, WT = np.meshgrid(wr, wt, indexing="ij")

  x = Rgrid * np.cos(Tgrid)
  y = Rgrid * np.sin(Tgrid)
  w = WR * WT

  interior_x = jnp.array(np.stack([x.ravel(), y.ravel()], axis=1))
  interior_w = jnp.array(w.ravel())

  # Boundary (circle r=R)
  xb = R * np.cos(theta)
  yb = R * np.sin(theta)
  wb = R * wt  # arc length = R dθ

  boundary_x = jnp.array(np.stack([xb, yb], axis=1))
  boundary_w = jnp.array(wb)

  tangent = jnp.array(np.stack([-np.sin(theta), np.cos(theta)], axis=1))
  normal  = jnp.array(np.stack([ np.cos(theta), np.sin(theta)], axis=1))

  return Quadrature(
    dim=2,
    shape=(nr, nt),
    interior_x=interior_x,
    interior_w=interior_w,
    boundary_x=boundary_x,
    boundary_w=boundary_w,
    boundary_tangent=tangent,
    boundary_normal=normal
  )


# =====================
# Domain Decomposition
# =====================

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DDQuadrature(Quadrature):
  """
  Quadrature with DD metadata for 1D (extends your Quadrature).
  - subdomain_id: integer id of this subdomain
  - neighbor_ids: (Jn,) tuple of neighbor subdomain ids (order defines columns)
  - owner_j_at_bndry: (Nb,) int32; neighbor id that owns that boundary point; -1 if global boundary
  - boundary_owner_onehot: (Nb, Jn) float32; one-hot over neighbor_ids; zeros row on global boundary
  - boundary_mask_global: (Nb,) bool; True where boundary point is a global boundary
  """
  subdomain_id: int
  neighbor_ids: Tuple[int, ...]
  owner_j_at_bndry: jax.Array          # (Nb,), int32; -1 on global boundary
  boundary_owner_onehot: jax.Array     # (Nb, Jn), float32 one-hot; zeros on global boundary
  boundary_mask_global: jax.Array      # (Nb,), bool


def _owner_onehot_from_ids(
  owner_j_at_bndry: jax.Array,
  neighbor_ids: Tuple[int, ...]
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


def dd_overlapping_interval_quadratures(
  bounds: Tuple[float, float] = (0.0, 1.0),
  mid: float = 0.5,
  overlap: float = 0.2,
  ng: int = 64,
):
  """
  Build two overlapping subdomain quadratures on [a,b]:
    sub 0: [a, mid + overlap/2]
    sub 1: [mid - overlap/2, b]
  Each DDQuadrature has neighbor_ids, owner_j_at_bndry, boundary_owner_onehot, boundary_mask_global.
  """
  a, b = bounds
  assert b > a
  assert 0.0 < overlap < (b - a)
  left_b  = float(jnp.clip(mid + 0.5 * overlap, a, b))
  right_a = float(jnp.clip(mid - 0.5 * overlap, a, b))
  assert right_a < left_b, "Invalid overlap: subdomains do not overlap."

  # Base quadratures for each sub-interval (assumes your builder yields 2 boundary points [left,right])
  q0 = gauss_legendre_interval_quadrature((a, left_b), ng)   # boundary_x ~ [[a],[left_b]]
  q1 = gauss_legendre_interval_quadrature((right_a, b), ng)  # boundary_x ~ [[right_a],[b]]

  # --- Subdomain 0 metadata ---
  sub0_id      = 0
  sub0_neigh   = (1,)  # columns order for one-hot
  # Ownership: left boundary is global, right boundary owned by subdomain 1
  sub0_owner   = jnp.array([-1, 1], dtype=jnp.int32)         # (Nb=2,)
  sub0_onehot  = _owner_onehot_from_ids(sub0_owner, sub0_neigh)  # (2,1)
  sub0_mask_g  = (sub0_owner == -1)                           # (2,)

  Q0 = DDQuadrature(
    # inherited Quadrature fields
    dim=q0.dim,
    shape=q0.shape,
    interior_x=q0.interior_x,
    interior_w=q0.interior_w,
    boundary_x=q0.boundary_x,
    boundary_w=q0.boundary_w,
    boundary_tangent=q0.boundary_tangent,
    boundary_normal=q0.boundary_normal,
    # DD fields
    subdomain_id=sub0_id,
    neighbor_ids=sub0_neigh,
    owner_j_at_bndry=sub0_owner,
    boundary_owner_onehot=sub0_onehot,
    boundary_mask_global=sub0_mask_g,
  )

  # --- Subdomain 1 metadata ---
  sub1_id      = 1
  sub1_neigh   = (0,)
  # Ownership: left boundary owned by subdomain 0, right boundary is global
  sub1_owner   = jnp.array([0, -1], dtype=jnp.int32)
  sub1_onehot  = _owner_onehot_from_ids(sub1_owner, sub1_neigh)
  sub1_mask_g  = (sub1_owner == -1)

  Q1 = DDQuadrature(
    dim=q1.dim,
    shape=q1.shape,
    interior_x=q1.interior_x,
    interior_w=q1.interior_w,
    boundary_x=q1.boundary_x,
    boundary_w=q1.boundary_w,
    boundary_tangent=q1.boundary_tangent,
    boundary_normal=q1.boundary_normal,
    subdomain_id=sub1_id,
    neighbor_ids=sub1_neigh,
    owner_j_at_bndry=sub1_owner,
    boundary_owner_onehot=sub1_onehot,
    boundary_mask_global=sub1_mask_g,
  )

  return Q0, Q1