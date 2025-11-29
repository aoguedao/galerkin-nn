# %%
"""
2D Poisson with Robin BC on a rectangle using four overlapping rectangular
subdomains (ASM/Robin-Schwarz). Manufactured solution:
  u(x,y) = sin(2 pi x) sin(2 pi y)
Boundary condition: κ ∂ₙ u + ε^{-1} u = h.
"""

import time
from typing import Callable, List, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import struct

from galerkinnn import FunctionState, DDPDE, GalerkinNN
from galerkinnn.quadratures import dd_overlapping_rectangle_four_quadratures
from galerkinnn.utils import make_u_fn, compare_num_exact_2d

# ----------------------------
# Parameters
# ----------------------------
bounds = ((0.0, 1.0), (0.0, 1.0))
midx, midy = 0.5, 0.5
overlapx, overlapy = 0.1, 0.1
nx, ny = 96, 96
n_edge = 96

eps_phys = 1e-1          # physical Robin ε on the outer boundary
eps_interface = 1e-4     # interface impedance parameter ε_Γ
kappa = 1.0

max_sweeps = 4
omega = 1.0

max_bases = 6
max_epoch_basis = 40
seeds = [42, 43, 44, 45]

N = 200      # base width per basis
r = 2       # width growth factor
A = 1e-3    # initial learning rate
rho = 1.1   # LR decay per basis


# ----------------------------
# Network
# ----------------------------
def net_fn(
  X: jax.Array,
  params: optax.Params,
  activation: Callable[[jax.Array], jax.Array],
) -> jax.Array:
  return activation(jnp.dot(X, params["W"]) + params["b"])


def activations_fn(i: int):
  scale = float(i)
  return lambda x: jnp.tanh(scale * x)


network_widths_fn = lambda i: int(min(N * (r ** (i - 1)), 512))
learning_rates_fn = lambda i: A * (rho ** (-(i - 1)))


# ----------------------------
# Helpers
# ----------------------------
def as_coeff_vector(u_coeff):
  if isinstance(u_coeff, (list, tuple)):
    return jnp.array([jnp.asarray(c).reshape(-1)[0] for c in u_coeff])
  c = jnp.asarray(u_coeff)
  return c.reshape(-1)


def make_grad2d(u_fn):
  """Return grad_u(X) with shape (N,2)."""
  def scalar_fn(xy):
    return u_fn(xy.reshape(1, 2))[0, 0]
  grad_scalar = jax.grad(scalar_fn)

  @jax.jit
  def grad(X):
    X = jnp.asarray(X).reshape(-1, 2)
    return jax.vmap(grad_scalar)(X)
  return grad


def relax_fn(new_fn, old_fn, omega_val: float):
  if omega_val == 1.0:
    return new_fn
  return lambda X: (1 - omega_val) * old_fn(X) + omega_val * new_fn(X)


def rectangle_normal(
  X: jax.Array,
  bounds_xy: tuple[tuple[float, float], tuple[float, float]],
  tol: float = 1e-7,
) -> jax.Array:
  (ax, bx), (ay, by) = bounds_xy
  X = jnp.asarray(X).reshape(-1, 2)
  x = X[:, 0:1]
  y = X[:, 1:2]

  nx = jnp.zeros_like(x)
  ny = jnp.zeros_like(y)

  nx = jnp.where(jnp.isclose(x, ax, atol=tol), -1.0, nx)
  nx = jnp.where(jnp.isclose(x, bx, atol=tol),  1.0, nx)
  ny = jnp.where(jnp.isclose(y, ay, atol=tol), -1.0, ny)
  ny = jnp.where(jnp.isclose(y, by, atol=tol),  1.0, ny)

  n = jnp.concatenate([nx, ny], axis=1)
  norm = jnp.sqrt(jnp.sum(n ** 2, axis=1, keepdims=True))
  norm = jnp.maximum(norm, jnp.array(1e-12, norm.dtype))
  return n / norm


# ---------------------------------------------
# Subdomain PDE: κ=const, physical Robin on ∂Ω
# ---------------------------------------------
@struct.dataclass
class Poisson2DRobinConstK:
  kappa: float = 1.0
  eps: float = 1e-2

  def exact_solution(self):
    two_pi = jnp.array(2.0 * jnp.pi, dtype=jnp.float32)

    def u_exact(X: jax.Array) -> jax.Array:
      X = jnp.asarray(X)
      x, y = X[:, 0], X[:, 1]
      val = jnp.sin(two_pi * x) * jnp.sin(two_pi * y)
      return val.reshape(-1, 1)
    return u_exact

  def source(self):
    kappa = jnp.asarray(self.kappa, dtype=jnp.float32)
    two_pi = jnp.array(2.0 * jnp.pi, dtype=jnp.float32)

    def f(X: jax.Array) -> jax.Array:
      X = jnp.asarray(X)
      x, y = X[:, 0], X[:, 1]
      val = 2.0 * (two_pi ** 2) * kappa * jnp.sin(two_pi * x) * jnp.sin(two_pi * y)
      return val.reshape(-1, 1)
    return f

  def boundary_data(self):
    kappa = jnp.asarray(self.kappa, dtype=jnp.float32)
    inv_eps = jnp.asarray(1.0 / self.eps, dtype=jnp.float32)
    two_pi = jnp.array(2.0 * jnp.pi, dtype=jnp.float32)
    tol = jnp.array(1e-7, dtype=jnp.float32)

    def h(B: jax.Array) -> jax.Array:
      B = jnp.asarray(B)
      x, y = B[:, 0], B[:, 1]
      left = x <= tol
      right = x >= (1.0 - tol)
      bottom = y <= tol
      top = y >= (1.0 - tol)

      sin_2pix = jnp.sin(two_pi * x)
      sin_2piy = jnp.sin(two_pi * y)
      cos_2pix = jnp.cos(two_pi * x)
      cos_2piy = jnp.cos(two_pi * y)

      u_bnd = sin_2pix * sin_2piy
      du_dx = two_pi * cos_2pix * sin_2piy
      du_dy = two_pi * sin_2pix * cos_2piy

      h_left = kappa * (-du_dx) + inv_eps * u_bnd
      h_right = kappa * (du_dx) + inv_eps * u_bnd
      h_bottom = kappa * (-du_dy) + inv_eps * u_bnd
      h_top = kappa * (du_dy) + inv_eps * u_bnd

      edge_val = jnp.zeros_like(x)
      edge_val = jnp.where(left, h_left, edge_val)
      edge_val = jnp.where(right, h_right, edge_val)
      edge_val = jnp.where(bottom, h_bottom, edge_val)
      edge_val = jnp.where(top, h_top, edge_val)
      return edge_val.reshape(-1, 1)
    return h

  def linear_operator(self):
    f = self.source()
    h = self.boundary_data()

    def L(v: FunctionState, quad) -> jax.Array:
      fvals = f(quad.interior_x)
      Li = jnp.einsum("n,ni->i", quad.interior_w, (fvals * v.interior)[:, :])
      hvals = h(quad.boundary_x)
      Lb = jnp.einsum("n,ni->i", quad.boundary_w, (hvals * v.boundary)[:, :])
      return (Li + Lb).reshape(1, -1)
    return L

  def bilinear_form(self):
    kappa = self.kappa
    inv_eps = 1.0 / self.eps

    def a(u: FunctionState, v: FunctionState, quad) -> jax.Array:
      a_grad = jnp.einsum(
        "nui,nvi,n->uv",
        u.grad_interior, v.grad_interior, quad.interior_w
      )
      a_bnd = jnp.einsum(
        "an,am,a->nm",
        u.boundary, v.boundary, quad.boundary_w
      )
      return kappa * a_grad + inv_eps * a_bnd
    return a

  def energy_norm(self):
    kappa = self.kappa
    inv_eps = 1.0 / self.eps

    def norm(v: FunctionState, quad) -> jax.Array:
      grad_sq = jnp.sum(v.grad_interior ** 2, axis=2)
      a1 = jnp.einsum("n,ni->i", quad.interior_w, grad_sq)
      b_sq = v.boundary ** 2
      a2 = jnp.einsum("n,ni->i", quad.boundary_w, b_sq)
      en2 = kappa * a1 + inv_eps * a2
      en2 = jnp.maximum(en2, jnp.array(0.0, en2.dtype))
      return jnp.sqrt(en2)
    return norm


# ===========================
# ======== MAIN ========
# ===========================
def build_pou_weights_rect4(Qs: Tuple) -> List[Callable[[jax.Array], jax.Array]]:
  """Indicator-based normalized weights over four overlapping rectangles."""
  rect_bounds = [Q.meta["bounds"] for Q in Qs]

  def weight_fn(bounds_xy):
    (ax, bx), (ay, by) = bounds_xy
    def w(X):
      X = jnp.asarray(X).reshape(-1, 2)
      x = X[:, 0]
      y = X[:, 1]
      inside = (x >= ax) & (x <= bx) & (y >= ay) & (y <= by)
      return inside.astype(jnp.float64).reshape(-1, 1)
    return w

  raw_w = [weight_fn(b) for b in rect_bounds]

  def normalize_ws(X):
    vals = [w(X) for w in raw_w]
    stack = jnp.hstack(vals)
    denom = jnp.sum(stack, axis=1, keepdims=True)
    denom = jnp.maximum(denom, jnp.array(1e-12, denom.dtype))
    return [v / denom for v in vals]

  def make_weight_i(i):
    def wi(X):
      return normalize_ws(X)[i]
    return wi

  return [make_weight_i(i) for i in range(4)]


def make_trace_fn(u_fn, grad_fn, bounds_xy):
  def g(X, grad_fn_inner=grad_fn, bounds_inner=bounds_xy):
    X = jnp.asarray(X)
    n = rectangle_normal(X, bounds_inner)
    grad = grad_fn_inner(X)
    n_dot_grad = jnp.sum(grad * n, axis=1, keepdims=True)
    return u_fn(X) + eps_interface * (kappa * n_dot_grad)
  return g


Qs = dd_overlapping_rectangle_four_quadratures(
  bounds=bounds,
  midx=midx,
  midy=midy,
  overlapx=overlapx,
  overlapy=overlapy,
  nx=nx,
  ny=ny,
  n_edge=n_edge,
)

for i, Q in enumerate(Qs):
  print(f"Subdomain {i}: n_int={Q.n_interior}, n_bnd={Q.n_boundary}, neighbors={Q.neighbor_ids}")

pde_base = Poisson2DRobinConstK(kappa=kappa, eps=eps_phys)
u_exact_fn = pde_base.exact_solution()

z = lambda X: jnp.zeros((X.shape[0], 1))
gradz = lambda X: jnp.zeros((X.shape[0], 2))
u_states = [FunctionState.from_function(z, Q, gradz) for Q in Qs]
u_fns = [z for _ in Qs]
grad_fns = [gradz for _ in Qs]

rect_bounds = [Q.meta["bounds"] for Q in Qs]

# one trace function per neighbor, aligned with neighbor_ids ordering
g_prev: List[List[Callable[[jax.Array], jax.Array]]] = [
  [z for _ in Q.neighbor_ids] for Q in Qs
]

start = time.perf_counter()

for k in range(max_sweeps):
  print(f"===== Schwarz sweep {k + 1} / {max_sweeps} =====")

  for i, Q in enumerate(Qs):
    neigh_ids = Q.neighbor_ids
    trace_fns = []
    for j, nid in enumerate(neigh_ids):
      g_new = make_trace_fn(u_fns[nid], grad_fns[nid], rect_bounds[nid])
      g_relaxed = relax_fn(g_new, g_prev[i][j], omega)
      g_prev[i][j] = g_relaxed
      trace_fns.append(g_relaxed)

    pde = DDPDE(base=pde_base, eps_interface=eps_interface, trace_fns=tuple(trace_fns))
    out = GalerkinNN(pde, Q).solve(
      seed=seeds[i] + 100 * k,
      u0=u_states[i],
      net_fn=net_fn,
      activations_fn=activations_fn,
      network_widths_fn=network_widths_fn,
      learning_rates_fn=learning_rates_fn,
      max_bases=max_bases,
      max_epoch_basis=max_epoch_basis,
      tol_solution=1e-8,
      tol_basis=1e-6,
    )
    u_state_out, u_coeff, *_rest, basis_coeff_list, sigma_list = out
    u_states[i] = u_state_out  # warm-start next sweep
    u_coeff_vec = as_coeff_vector(u_coeff)
    u_fns[i] = make_u_fn(sigma_list, u_coeff_vec, basis_coeff_list)
    grad_fns[i] = make_grad2d(u_fns[i])

    print(f"  Subdomain {i} solve done.")

elapsed = time.perf_counter() - start
print(f"Total elapsed time: {elapsed:.3f} s")

# -----------------------
# Stitch global solution
# -----------------------
weights = build_pou_weights_rect4(Qs)

def u_global_fn(X):
  X = jnp.asarray(X).reshape(-1, 2)
  wvals = [w(X) for w in weights]
  parts = [wvals[i] * u_fns[i](X) for i in range(4)]
  return sum(parts)

# -----------------------
# Error & visualization
# -----------------------
Nx = 200
Ny = 200
xg = jnp.linspace(bounds[0][0], bounds[0][1], Nx)
yg = jnp.linspace(bounds[1][0], bounds[1][1], Ny)
Xg, Yg = jnp.meshgrid(xg, yg, indexing="ij")
X_flat = jnp.stack([Xg.ravel(), Yg.ravel()], axis=1)

compare_num_exact_2d(
  X_flat,
  u_num_fn=lambda X: u_global_fn(X),
  u_exact_fn=lambda X: u_exact_fn(X),
  kind="tri",
  titles=("Exact", "Numerical", "Abs error"),
  error_kind="absolute",
)
