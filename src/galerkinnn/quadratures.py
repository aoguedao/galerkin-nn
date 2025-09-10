import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple
from dataclasses import dataclass
from scipy.special import roots_jacobi, eval_jacobi


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Quadrature:
  ng: int
  dim: int
  interior_x: jax.Array      # (N_interior, dim)
  interior_w: jax.Array      # (N_interior,)
  boundary_x: jax.Array      # (N_boundary, dim)
  boundary_w: jax.Array      # (N_boundary,)
  boundary_tangent: jax.Array  # (N_boundary, dim)
  boundary_normal: jax.Array   # (N_boundary, dim)

  def integrate_interior(self, values: jax.Array) -> jax.Array:
    # if values.ndim == 1:
    #   return jnp.sum(values * self.interior_w)
    # else:
    #   return jnp.sum(values * self.interior_w[:, None], axis=0)
    return jnp.sum(values * self.interior_w[:, None], axis=0, keepdims=True)

  def integrate_boundary(self, values: jax.Array) -> jax.Array:
    # if values.ndim == 1:
    #   return jnp.sum(values * self.boundary_w)
    # else:
    #   return jnp.sum(values * self.boundary_w[:, None], axis=0)
    return jnp.sum(values * self.boundary_w[:, None], axis=0, keepdims=True)


def gauss_lobatto(ng: int):
  if ng < 2:
    raise ValueError("Gauss-Lobatto requires ng >= 2")
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
  # boundary_w = jnp.array([w_mapped[0], w_mapped[-1]])
  boundary_w = jnp.array([1.0, 1.0])
  boundary_tangent = jnp.array([[1.0], [1.0]])
  boundary_normal = jnp.array([[-1.0], [1.0]])

  return Quadrature(
    ng,
    1,
    interior_x,
    interior_w,
    boundary_x,
    boundary_w,
    boundary_tangent,
    boundary_normal
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


def gauss_lobatto_legendre_circular_sector_quadrature(ng: int, theta: float) -> Quadrature:
  nr, nt = ng, max(2, int(np.ceil(ng * theta)))
  xr, wr = gauss_lobatto(nr)
  r_nodes = 0.5 * (xr + 1.0)
  wr_scaled = 0.5 * wr
  xt, wt = roots_jacobi(nt, 0.0, 0.0)
  t_nodes = 0.5 * theta * (xt + 1.0)
  wt_scaled = 0.5 * theta * wt

  R, T = jnp.meshgrid(jnp.array(r_nodes), jnp.array(t_nodes), indexing='ij')
  X = (R * jnp.cos(T)).ravel()
  Y = (R * jnp.sin(T)).ravel()
  interior_x = jnp.stack([X, Y], axis=1)
  interior_w = (jnp.outer(wr_scaled, wt_scaled) * R).ravel()

  boundary_points, boundary_weights, tangents, normals = [], [], [], []

  r = jnp.array(r_nodes)
  # θ=0 edge
  boundary_points.append(jnp.stack([r, jnp.zeros_like(r)], axis=1))
  boundary_weights.append(wr_scaled)
  tangents.append(jnp.stack([jnp.ones_like(r), jnp.zeros_like(r)], axis=1))
  normals.append(jnp.stack([jnp.zeros_like(r), -jnp.ones_like(r)], axis=1))
  # θ=theta edge
  boundary_points.append(jnp.stack([r * jnp.cos(theta), r * jnp.sin(theta)], axis=1))
  boundary_weights.append(wr_scaled)
  tangents.append(jnp.stack([jnp.full_like(r, jnp.cos(theta)),
                             jnp.full_like(r, jnp.sin(theta))], axis=1))
  normals.append(jnp.stack([-jnp.full_like(r, jnp.sin(theta)),
                            jnp.full_like(r, jnp.cos(theta))], axis=1))
  # arc r=1
  t = jnp.array(t_nodes)
  boundary_points.append(jnp.stack([jnp.cos(t), jnp.sin(t)], axis=1))
  boundary_weights.append(wt_scaled)
  tangents.append(jnp.stack([-jnp.sin(t), jnp.cos(t)], axis=1))
  normals.append(jnp.stack([jnp.cos(t), jnp.sin(t)], axis=1))

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
