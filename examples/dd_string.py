# %%
"""
DEBUG: 1D Poisson with Robin BC and known analytic solution.
Problem:
  -u'' = f  on (0,1),  u + eps * ∂_n u = 0 on {0,1},  eps = 1e-3
  f(x) = (2π)^2 sin(2πx) + (4π)^2 sin(4πx) + (6π)^2 sin(6πx)
  u*(x) = sin(2πx)+sin(4πx)+sin(6πx) + [(-24π ε) x + 12π ε] / (1 + 2 ε)

Runs Robin–Schwarz ASM with two overlapping 1D subdomains (κ=1 on both).
"""

from typing import Callable, List

import jax
import jax.numpy as jnp
from flax import struct

from galerkinnn import FunctionState, DDPDE, GalerkinNN
from galerkinnn.quadratures import dd_overlapping_interval_quadratures
from galerkinnn.pou import build_pou_weights_1d
from galerkinnn.utils import make_u_fn

# ----------------------------
# Parameters
# ----------------------------

# Quadratures
bounds = (0.0, 1.0)
mid = 1 / 4
overlap = 0.25
ng = 128
Q0, Q1 = dd_overlapping_interval_quadratures(bounds=bounds, mid=mid, overlap=overlap, n_interior=ng)

eps_phys= 1e-3          # physical Robin ε
eps_interface = 5e-4          # interface ε_Γ
k_left, k_right = 1.0, 1.0     # uniform medium
max_sweeps = 8
tol_jump = 5e-5
omega = 1
seed0, seed1 = 42, 43

N   = 5      # base width per basis
r   = 2      # growth factor per basis
A   = 5e-3   # initial LR
rho = 1.1    # LR decay per basis
max_epoch_basis = 100
max_bases_0, max_bases_1 = 3, 4

# ----------------------------
# Network & schedules
# ----------------------------
def net_fn(X, params, activation):
  return activation(X @ params["W"] + params["b"])

def activations_fn(i: int):
  scale_fn = lambda i: i
  s = scale_fn(i)
  return lambda x: jnp.tanh(s * x)

network_widths_fn = lambda i: N * (r ** (i - 1))
learning_rates_fn = lambda i: A * (rho ** (-(i - 1)))

# ----------------------------
# Helpers
# ----------------------------
def as_coeff_vector(u_coeff):
  if isinstance(u_coeff, (list, tuple)):
    return jnp.array([jnp.asarray(c).reshape(-1)[0] for c in u_coeff])
  c = jnp.asarray(u_coeff)
  return c.reshape(-1)

def make_grad1d(u_fn):
  """grad_u(X): (N,1); compute pointwise scalar grad and vmap."""
  def f_scalar(x):
    return u_fn(x.reshape(1,1))[0,0]
  g_scalar = jax.grad(f_scalar)
  return lambda X: jax.vmap(g_scalar)(X.reshape(-1)).reshape(-1,1)

def relax_fn(new_fn, old_fn, omega: float):
  if omega == 1.0:
    return new_fn
  return lambda X: (1 - omega) * old_fn(X) + omega * new_fn(X)

# ---------------------------------------------
# Subdomain PDE: κ=const, physical Robin on ∂Ω
# ---------------------------------------------
@struct.dataclass
class Poisson1DRobinConstK:
  k: float = 1.0       # constant κ on this subdomain
  eps: float = 1e-3    # physical Robin parameter ε

  def source(self):
    # -u'' = f, with the f from the problem
    def f(X: jax.Array) -> jax.Array:
      x = X.reshape(-1, 1)
      return ((2*jnp.pi)**2 * jnp.sin(2*jnp.pi*x)
            + (4*jnp.pi)**2 * jnp.sin(4*jnp.pi*x)
            + (6*jnp.pi)**2 * jnp.sin(6*jnp.pi*x))
    return f

  def bilinear_form(self):
    k = self.k; eps = self.eps
    def a(u: FunctionState, v: FunctionState, quad) -> jax.Array:
      Avol = jnp.einsum("nui,nvi,n->uv", u.grad_interior, v.grad_interior, quad.interior_w * k)
      gamma_b = (1.0 / eps) * quad.boundary_w
      mask_g  = getattr(quad, "boundary_mask_global", None)
      if mask_g is not None:
        gamma_b = gamma_b * mask_g.astype(gamma_b.dtype)  # physical boundary only
      Abnd = jnp.einsum("an,am,a->nm", u.boundary, v.boundary, gamma_b)
      return Avol + Abnd
    return a

  def linear_operator(self):
    f = self.source()
    def L(v: FunctionState, quad) -> jax.Array:
      fv = f(quad.interior_x).reshape(-1)
      return jnp.einsum("n,nv,n->v", fv, v.interior, quad.interior_w)
    return L

  def energy_norm(self):
    k = self.k; eps = self.eps
    def en(v: FunctionState, quad) -> jax.Array:
      e1 = jnp.einsum("n,ni->i", quad.interior_w * k, jnp.sum(v.grad_interior**2, axis=2))
      gamma_b = (1.0 / eps) * quad.boundary_w
      mask_g  = getattr(quad, "boundary_mask_global", None)
      if mask_g is not None:
        gamma_b = gamma_b * mask_g.astype(gamma_b.dtype)
      e2 = jnp.einsum("n,ni->i", gamma_b, v.boundary**2)
      return jnp.sqrt(jnp.maximum(e1 + e2, 0.0))
    return en

# ===========================
# ======== MAIN BODY ========
# ===========================
# PDEs
pde_left  = Poisson1DRobinConstK(k=k_left,  eps=eps_phys)
pde_right = Poisson1DRobinConstK(k=k_right, eps=eps_phys)

# PoU weights for stitching
w0_fn, w1_fn = build_pou_weights_1d(Q0, Q1)

# Zero states & traces
z     = lambda X: jnp.zeros((X.shape[0], 1))
gradz = lambda X: jnp.zeros((X.shape[0], 1))
u0_state = FunctionState.from_function(z, Q0, gradz)
u1_state = FunctionState.from_function(z, Q1, gradz)
g0 = z  # data fed into Ω0 (from right)
g1 = z  # data fed into Ω1 (from left)

history = []
u0_fn = z
u1_fn = z

for k in range(max_sweeps):
  print(f"===== Sweep: {k+ 1} =====")
  # ---------- Ω0 (LEFT) ----------
  pde0 = DDPDE(base=pde_left, eps_interface=eps_interface, trace_fns=(g0,))
  out0 = GalerkinNN(pde0, Q0).solve(
    seed=seed0 + 100*k,
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
  u0_coeff_vec = as_coeff_vector(u0_coeff)
  u0_fn = make_u_fn(sigma_list0, u0_coeff_vec, basis_coeff_list0)
  grad_u0_fn = make_grad1d(u0_fn)

  # data for Ω1: g = u_L + ε_Γ (n_L·κ_L u'_L), n_L = +1
  g1_new = lambda X: u0_fn(X) + eps_interface * (k_left * grad_u0_fn(X))
  g1 = relax_fn(g1_new, g1, omega)

  # ---------- Ω1 (RIGHT) ----------
  pde1 = DDPDE(base=pde_right, eps_interface=eps_interface, trace_fns=(g1,))
  out1 = GalerkinNN(pde1, Q1).solve(
    seed=seed1 + 100*k,
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
  u1_coeff_vec = as_coeff_vector(u1_coeff)
  u1_fn = make_u_fn(sigma_list1, u1_coeff_vec, basis_coeff_list1)
  grad_u1_fn = make_grad1d(u1_fn)

  # data for Ω0: g = u_R + ε_Γ (n_R·κ_R u'_R), n_R = −1
  g0_new = lambda X: u1_fn(X) + eps_interface * (-k_right * grad_u1_fn(X))
  g0 = relax_fn(g0_new, g0, omega)

  # diagnostics
  x_if_R = Q0.boundary_x[-1:]  # b0^+
  x_if_L = Q1.boundary_x[:1]   # a1^-
  u0_if = jnp.squeeze(u0_fn(x_if_R)).item()
  u1_if = jnp.squeeze(u1_fn(x_if_L)).item()
  jump = u0_if - u1_if
  print(f"[sweep {k+1:02d}] u0(b0^+)={u0_if:+.6e}, u1(a1^-)={u1_if:+.6e}, jump={jump:+.6e}")
  history.append(dict(sweep=k+1, u0_if=u0_if, u1_if=u1_if, jump=jump))
  if abs(jump) < tol_jump:
    break

# Stitched global solution via PoU
u_glob_fn = lambda X: w0_fn(X) * u0_fn(X) + w1_fn(X) * u1_fn(X)

# -----------------------
# Sanity checks & plots
# -----------------------
# Flux jump (should be small)
xR = Q0.boundary_x[-1:].reshape(1,1)   # b0^+
xL = Q1.boundary_x[:1].reshape(1,1)    # a1^-
qL = k_left  * grad_u0_fn(xR)[0,0]
qR = k_right * grad_u1_fn(xL)[0,0]
J_q = float(abs(qL - qR))

# Physical Robin residuals (u + eps * ∂_n u = 0)
x0L = Q0.boundary_x[:1].reshape(1,1)   # left end, n = -1 ⇒ u - eps*u' = 0
x1R = Q1.boundary_x[-1:].reshape(1,1)  # right end, n = +1 ⇒ u + eps*u' = 0
R_L = float(u0_fn(x0L)[0,0] - eps_phys * grad_u0_fn(x0L)[0,0])
R_R = float(u1_fn(x1R)[0,0] + eps_phys * grad_u1_fn(x1R)[0,0])
print(f"[sanity] flux jump |k u0' - k u1'| = {J_q:.3e}")
print(f"[sanity] |Robin left|={abs(R_L):.3e}, |Robin right|={abs(R_R):.3e}")

# Strong-form residuals in interiors (||r||_L2)
def strong_residual(u_fn, grad_u_fn, kappa, X):
  # r = -d/dx (k u') - f = -k u'' - f; here k=1 and f is the PDE source
  # compute u'' by differentiating grad_u pointwise
  def g_scalar(x):
    return grad_u_fn(x.reshape(1,1))[0,0]
  uxx = jax.vmap(jax.grad(g_scalar))(X.reshape(-1)).reshape(-1,1)
  # f from the PDE definition:
  x = X.reshape(-1,1)
  f = ((2*jnp.pi)**2 * jnp.sin(2*jnp.pi*x)
     + (4*jnp.pi)**2 * jnp.sin(4*jnp.pi*x)
     + (6*jnp.pi)**2 * jnp.sin(6*jnp.pi*x))
  return -kappa * uxx - f

X0 = Q0.interior_x
X1 = Q1.interior_x
r0 = strong_residual(u0_fn, grad_u0_fn, k_left,  X0)
r1 = strong_residual(u1_fn, grad_u1_fn, k_right, X1)
L2r0 = float(jnp.sqrt(jnp.sum((r0**2)*Q0.interior_w[:,None])))
L2r1 = float(jnp.sqrt(jnp.sum((r1**2)*Q1.interior_w[:,None])))
print(f"[sanity] ||r||_L2(Ω0) ≈ {L2r0:.3e},  ||r||_L2(Ω1) ≈ {L2r1:.3e}")

# Analytic solution
def u_exact_robin_sines(eps: float):
  c = 1.0 / (1.0 + 2.0*eps)
  def u(X: jax.Array) -> jax.Array:
    x = X.reshape(-1,1)
    return (jnp.sin(2*jnp.pi*x) + jnp.sin(4*jnp.pi*x) + jnp.sin(6*jnp.pi*x)
            + c * (-24.0*jnp.pi*eps * x + 12.0*jnp.pi*eps))
  return u

import numpy as np
import matplotlib.pyplot as plt

# Overlap shading
a0, b0 = float(Q0.boundary_x[0,0]), float(Q0.boundary_x[-1,0])
a1, b1 = float(Q1.boundary_x[0,0]), float(Q1.boundary_x[-1,0])
a_ov, b_ov = max(a0, a1), min(b0, b1)

# Evaluate
x_plot = np.linspace(bounds[0], bounds[1], 801).reshape(-1,1)
u_num  = np.asarray(u_glob_fn(x_plot)).ravel()
u_ex   = np.asarray(u_exact_robin_sines(eps_phys)(x_plot)).ravel()

# Errors
Linf = float(np.max(np.abs(u_num - u_ex)))
L2   = float(np.sqrt(np.trapezoid((u_num - u_ex)**2, x_plot.ravel())))
print(f"[analytic] L∞ = {Linf:.3e},  L2 = {L2:.3e}")

# Plot
fig, ax = plt.subplots(figsize=(7.2,4))
if b_ov > a_ov:
  ax.axvspan(a_ov, b_ov, color="0.90", alpha=0.8, label="overlap", zorder=0)
ax.plot(x_plot, u_ex,  'k--', lw=2.5, label='Analytic')
ax.plot(x_plot, u_num, 'C0-',  lw=2, label='GalerkinNN-DD')
ax.axvline(mid, color='0.5', ls='--', lw=1)
ax.set_xlabel("$x$"); ax.set_ylabel("$u(x)$")
ax.set_title(rf"Poisson 1D: String Displacement ($\varepsilon$={eps_phys:g})")
ax.legend(); ax.grid(alpha=0.25)
fig.savefig(f"images/poisson1d_string_eps_{eps_phys}__maxbases_{max_bases_0},{max_bases_1}.png")
plt.tight_layout(); plt.show()

# # Geometry sanity print
# tol = 1e-12
# print("Q0.boundary_x =", np.array(Q0.boundary_x).ravel().tolist())
# print("Q1.boundary_x =", np.array(Q1.boundary_x).ravel().tolist())
# print("Q0.boundary_mask_global =", np.array(Q0.boundary_mask_global).astype(int).tolist())
# print("Q1.boundary_mask_global =", np.array(Q1.boundary_mask_global).astype(int).tolist())

# # %%
