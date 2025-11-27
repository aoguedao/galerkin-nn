# %%
"""
2D Poisson with Robin BC on a unit rectangle using a two-subdomain
Robin-Schwarz alternating method. Manufactured solution:
  u(x,y) = sin(pi x) sin(pi y) + c
Boundary condition: κ ∂ₙ u + ε^{-1} u = h.
"""

import time
from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import struct

from galerkinnn import FunctionState, DDPDE, GalerkinNN
from galerkinnn.quadratures import dd_overlapping_rectangle_quadratures, rectangle_quadrature
from galerkinnn.pou import build_pou_weights_rect
from galerkinnn.utils import make_u_fn, compare_num_exact_2d

# ----------------------------
# Parameters
# ----------------------------
bounds = ((0.0, 1.0), (0.0, 1.0))
mid = 0.5
overlap = 0.3
nx, ny = 128, 128
n_edge = 128

eps_phys = 1e-1          # physical Robin ε on the outer boundary
eps_interface = 1e-4     # interface impedance parameter ε_Γ
k_left, k_right = 1.0, 1.0

max_sweeps = 10
tol_jump = 1e-3
omega = 0.7

max_bases_0, max_bases_1 = 6, 6
max_epoch_basis = 50
seed0, seed1 = 42, 43

N = 8       # base width per basis
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


network_widths_fn = lambda i: int(min(N * (r ** (i - 1)), 1024))
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


def relax_fn(new_fn, old_fn, omega: float):
  if omega == 1.0:
    return new_fn
  return lambda X: (1 - omega) * old_fn(X) + omega * new_fn(X)

# ---------------------------------------------
# Subdomain PDE: κ=const, physical Robin on ∂Ω
# ---------------------------------------------
@struct.dataclass
class Poisson2DRobinConstK:
  kappa: float = 1.0
  eps: float = 1e-2
  c: float = 0.3
  tol_edge: float = 1e-7

  def exact_solution(self):
    c = jnp.asarray(self.c, dtype=jnp.float32)

    def u_exact(X: jax.Array) -> jax.Array:
      X = jnp.asarray(X)
      x, y = X[:, 0], X[:, 1]
      val = jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y) + c
      return val.reshape(-1, 1)
    return u_exact

  def source(self):
    kappa = jnp.asarray(self.kappa, dtype=jnp.float32)

    def f(X: jax.Array) -> jax.Array:
      X = jnp.asarray(X)
      x, y = X[:, 0], X[:, 1]
      val = 2.0 * kappa * (jnp.pi ** 2) * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)
      return val.reshape(-1, 1)
    return f

  def boundary_data(self):
    kappa = jnp.asarray(self.kappa, dtype=jnp.float32)
    inv_eps_c = jnp.asarray(self.c / self.eps, dtype=jnp.float32)
    tol = jnp.asarray(self.tol_edge, dtype=jnp.float32)

    def h(B: jax.Array) -> jax.Array:
      B = jnp.asarray(B)
      x, y = B[:, 0], B[:, 1]
      left = x <= tol
      right = x >= (1.0 - tol)
      bottom = y <= tol
      top = y >= (1.0 - tol)

      hy = -kappa * jnp.pi * jnp.sin(jnp.pi * y) + inv_eps_c
      hx = -kappa * jnp.pi * jnp.sin(jnp.pi * x) + inv_eps_c
      edge_val = jnp.zeros_like(x)
      edge_val = jnp.where(left | right, hy, edge_val)
      edge_val = jnp.where(bottom | top, hx, edge_val)
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
Q0, Q1 = dd_overlapping_rectangle_quadratures(bounds=bounds, mid=mid, overlap=overlap, nx=nx, ny=ny, n_edge=n_edge)
print(f"Subdomain 0 bounds ~ x ∈ [{float(np.min(Q0.boundary_x[:,0])):.3f}, {float(np.max(Q0.boundary_x[:,0])):.3f}]")
print(f"Subdomain 1 bounds ~ x ∈ [{float(np.min(Q1.boundary_x[:,0])):.3f}, {float(np.max(Q1.boundary_x[:,0])):.3f}]")

pde_left = Poisson2DRobinConstK(kappa=k_left, eps=eps_phys)
pde_right = Poisson2DRobinConstK(kappa=k_right, eps=eps_phys)

w0_fn, w1_fn = build_pou_weights_rect(Q0, Q1)

z = lambda X: jnp.zeros((X.shape[0], 1))
gradz = lambda X: jnp.zeros((X.shape[0], 2))
u0_state = FunctionState.from_function(z, Q0, gradz)
u1_state = FunctionState.from_function(z, Q1, gradz)

g0 = z
g1 = z

history = []
u0_fn = z
u1_fn = z

start = time.perf_counter()

for k in range(max_sweeps):
  print(f"===== Sweep: {k + 1} =====")
  # ---------- Ω0 (LEFT) ----------
  pde0 = DDPDE(base=pde_left, eps_interface=eps_interface, trace_fns=(g0,))
  out0 = GalerkinNN(pde0, Q0).solve(
    seed=seed0 + 100 * k,
    u0=u0_state,
    net_fn=net_fn,
    activations_fn=activations_fn,
    network_widths_fn=network_widths_fn,
    learning_rates_fn=learning_rates_fn,
    max_bases=max_bases_0,
    max_epoch_basis=max_epoch_basis,
    tol_solution=1e-8,
    tol_basis=1e-6,
  )
  u0_state_out, u0_coeff, *_rest0, basis_coeff_list0, sigma_list0 = out0
  u0_state = u0_state_out  # warm-start next sweep
  u0_coeff_vec = as_coeff_vector(u0_coeff)
  u0_fn = make_u_fn(sigma_list0, u0_coeff_vec, basis_coeff_list0)
  grad_u0_fn = make_grad2d(u0_fn)

  def g1_new(X, grad_fn=grad_u0_fn):
    grad = grad_fn(X)
    return u0_fn(X) + eps_interface * (k_left * grad[:, [0]])
  g1 = relax_fn(g1_new, g1, omega)

  # ---------- Ω1 (RIGHT) ----------
  pde1 = DDPDE(base=pde_right, eps_interface=eps_interface, trace_fns=(g1,))
  out1 = GalerkinNN(pde1, Q1).solve(
    seed=seed1 + 100 * k,
    u0=u1_state,
    net_fn=net_fn,
    activations_fn=activations_fn,
    network_widths_fn=network_widths_fn,
    learning_rates_fn=learning_rates_fn,
    max_bases=max_bases_1,
    max_epoch_basis=max_epoch_basis,
    tol_solution=1e-8,
    tol_basis=1e-6,
  )
  u1_state_out, u1_coeff, *_rest1, basis_coeff_list1, sigma_list1 = out1
  u1_state = u1_state_out  # warm-start next sweep
  u1_coeff_vec = as_coeff_vector(u1_coeff)
  u1_fn = make_u_fn(sigma_list1, u1_coeff_vec, basis_coeff_list1)
  grad_u1_fn = make_grad2d(u1_fn)

  def g0_new(X, grad_fn=grad_u1_fn):
    grad = grad_fn(X)
    return u1_fn(X) + eps_interface * (-k_right * grad[:, [0]])
  g0 = relax_fn(g0_new, g0, omega)

  # ---------- Diagnostics ----------
  mask_interface0 = ~np.asarray(Q0.boundary_mask_global)
  X_if = jnp.asarray(np.asarray(Q0.boundary_x)[mask_interface0])
  w_if = jnp.asarray(np.asarray(Q0.boundary_w)[mask_interface0])

  if X_if.shape[0] == 0:
    raise RuntimeError("Interface boundary for subdomain 0 is empty.")

  u0_if = jnp.squeeze(u0_fn(X_if))
  u1_if = jnp.squeeze(u1_fn(X_if))
  jump_u = u0_if - u1_if
  jump_max = float(jnp.max(jnp.abs(jump_u)))
  jump_L2 = float(jnp.sqrt(jnp.sum((jump_u ** 2) * w_if)))

  flux0 = k_left * grad_u0_fn(X_if)[:, 0]
  flux1 = -k_right * grad_u1_fn(X_if)[:, 0]
  flux_jump = flux0 + flux1
  flux_max = float(jnp.max(jnp.abs(flux_jump)))
  flux_L2 = float(jnp.sqrt(jnp.sum((flux_jump ** 2) * w_if)))

  print(f"[sweep {k + 1:02d}] max|jump(u)|={jump_max:.3e}, max|jump(q·n)|={flux_max:.3e}")
  print(f"[sweep {k + 1:02d}] ||jump(u)||_L2≈{jump_L2:.3e}, ||jump(q·n)||_L2≈{flux_L2:.3e}")

  history.append(dict(
    sweep=k + 1,
    jump_u_max=jump_max,
    jump_u_L2=jump_L2,
    jump_flux_max=flux_max,
    jump_flux_L2=flux_L2,
  ))

  if jump_max < tol_jump:
    print(f"Stopping: interface jump below tolerance ({tol_jump}).")
    break

elapsed = time.perf_counter() - start
print(f"Total elapsed time: {elapsed:.3f} s")

# %%
# Stitched global solution via PoU
u_glob_fn = lambda X: w0_fn(X) * u0_fn(X) + w1_fn(X) * u1_fn(X)

# -----------------------
# Diagnostics & plots
# -----------------------
exact_fn = pde_left.exact_solution()
quad_eval = rectangle_quadrature(bounds=bounds, nx=nx, ny=ny, n_edge=n_edge)
X_eval = quad_eval.interior_x
w_eval = quad_eval.interior_w

u_num = jnp.squeeze(u_glob_fn(X_eval))
u_exact = jnp.squeeze(exact_fn(X_eval))
err = u_num - u_exact

Linf = float(jnp.max(jnp.abs(err)))
L2 = float(jnp.sqrt(jnp.sum((err ** 2) * w_eval)))
print(f"[analytic] L∞ = {Linf:.3e},  L2 = {L2:.3e}")

# Physical Robin residuals on each subdomain
def robin_residual(pde: Poisson2DRobinConstK, u_fn, grad_fn, quad) -> float:
  mask = np.asarray(quad.boundary_mask_global)
  idx = np.where(mask)[0]
  if idx.size == 0:
    return 0.0
  X_b = jnp.asarray(np.asarray(quad.boundary_x)[idx])
  n_b = jnp.asarray(np.asarray(quad.boundary_normal)[idx])
  u_b = u_fn(X_b)
  grad_b = grad_fn(X_b)
  n_dot_grad = jnp.sum(grad_b * n_b, axis=1, keepdims=True)
  h_vals = pde.boundary_data()(X_b)
  residual = pde.kappa * n_dot_grad + (1.0 / pde.eps) * u_b - h_vals
  return float(jnp.max(jnp.abs(residual)))

R_left = robin_residual(pde_left, u0_fn, grad_u0_fn, Q0)
R_right = robin_residual(pde_right, u1_fn, grad_u1_fn, Q1)
print(f"[sanity] |Robin residual|_Ω0≈{R_left:.3e}, |Robin residual|_Ω1≈{R_right:.3e}")

# Transmission residuals (should match the set g-functions)
g1_now = g1_new(X_if)
g0_now = g0_new(X_if)
R_L_tr = float(jnp.max(jnp.abs(u0_fn(X_if) + eps_interface * (k_left * grad_u0_fn(X_if)[:, [0]]) - g1_now)))
R_R_tr = float(jnp.max(jnp.abs(u1_fn(X_if) + eps_interface * (-k_right * grad_u1_fn(X_if)[:, [0]]) - g0_now)))
print(f"[sanity] |Robin_trans_L(Γ)|≈{R_L_tr:.3e}, |Robin_trans_R(Γ)|≈{R_R_tr:.3e}")

# Plot numerical vs exact
fig, ax = compare_num_exact_2d(
  quad_eval.interior_x,
  u_glob_fn,
  exact_fn,
  kind="tri",
  error_kind="relative",
  savepath="images/poisson2d_rectangle_dd.png",
)
plt.show()

