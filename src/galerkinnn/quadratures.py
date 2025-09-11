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