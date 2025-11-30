# %%
"""
2D Poisson with κ = 1 on the union of a disk and a rectangle, solved by a
two-subdomain Robin–Schwarz alternating method with GalerkinNN.

Domain:
  Ω = Ω_disk ∪ Ω_rect
    Ω_disk = { (x,y): x^2 + y^2 < R^2 }
    Ω_rect = [ax, bx] × [ay, by].

Subdomains:
  Ω0 = Ω_disk      (DDQuadrature Q_disk)
  Ω1 = Ω_rect      (DDQuadrature Q_rect)

PDE (same on both subdomains):

  -Δu = f(x,y)        in Ω,
  u + eps_phys ∂_n u = h(x,y)   on ∂Ω_phys,

with manufactured exact solution:

  u(x,y) = cos(π x) cos(π y),
  f(x,y) = 2 π^2 cos(π x) cos(π y),
  h(x,y) = u(x,y) + eps_phys ∂_n u(x,y),

where n is the outer normal to the union domain. In the weak form we use:

  a(u,v) = ∫_Ω ∇u·∇v + eps_phys^{-1} ∫_{∂Ω_phys} u v,
  L(v)   = ∫_Ω f v + eps_phys^{-1} ∫_{∂Ω_phys} h v.

Interface Robin is added by DDPDE using eps_interface and trace_fns.
"""

from typing import Callable

import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax import struct

from galerkinnn import FunctionState, DDPDE, GalerkinNN
from galerkinnn.quadratures import (
  dd_overlapping_disk_rectangle_quadratures,
)
from galerkinnn.pou import build_pou_weights_disk_rect
from galerkinnn.utils import make_u_fn, as_coeff_vector, make_grad2d, relax_fn


EXPERIMENT = "dd_disk_rectangle"
stamp = datetime.now().strftime("%Y%m%d%H%M%S")
output_path = Path() / "output" / EXPERIMENT / stamp
images_path = output_path / "images"
images_path.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Parameters
# ----------------------------
R = 1.0
rect_bounds = ((0.0, 1.8), (-0.6, 0.6))

n_r = 128
n_theta = 128
nx_rect = 128
ny_rect = 128
n_edge_rect = None

kappa = 1.0
eps_phys = 1e-3       # physical Robin ε on global boundary
eps_interface = 1e-3  # interface impedance ε_Γ

max_sweeps = 6
omega = 1           # relaxation for interface trace updates

max_bases_0, max_bases_1 = 6, 6
max_epoch_basis = 50
seed0, seed1 = 42, 43

# network size / LR scheduling
# N = 8
# r_width = 2
# A = 5e-2
# rho = 1.1

N = 200  # Init Neurons
r_width = 100  # Neurons Growth
A = 5e-2  # Init Learning Rate
rho = 1.1  # Learning Rate Growth
max_neurons = 512

config = {
  "bounds": rect_bounds,
  "n_r": n_r,
  "n_theta": n_theta,
  "nx_rect": nx_rect,
  "ny_rect": ny_rect,
  "n_edge_rect": n_edge_rect,
  "eps_phys": eps_phys,
  "eps_interface": eps_interface,
  "kappa": kappa,
  "max_sweeps": max_sweeps,
  "omega": omega,
  "max_bases": [max_bases_0, max_bases_1],
  "max_epoch_basis": max_epoch_basis,
  "seeds": [seed0, seed1],
  "N": N,
  "r_width": r_width,
  "A": A,
  "rho": rho,
  "neuron_cap": max_neurons
}
(output_path / "config.json").write_text(json.dumps(config, indent=2))
print(config)

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


# network_widths_fn = lambda i: int(min(N * (r_width ** (i - 1)), 1024))
network_widths_fn = lambda i: int(min(N + r_width * int((i - 1) / 2), max_neurons))
learning_rates_fn = lambda i: A * (rho ** (-(i - 1)))


# ----------------------------
# Helpers
# ----------------------------
def radial_normal(X: jax.Array) -> jax.Array:
  X = jnp.asarray(X).reshape(-1, 2)
  r = jnp.sqrt(jnp.sum(X ** 2, axis=1, keepdims=True))
  denom = jnp.maximum(r, jnp.array(1e-12, r.dtype))
  return X / denom


def rectangle_normal(
  X: jax.Array,
  bounds: tuple[tuple[float, float], tuple[float, float]],
  tol: float = 1e-6,
) -> jax.Array:
  """
  Outward normal for the rectangle [ax,bx] x [ay,by].

  Assumes X is on (or close to) the rectangle boundary.
  """
  (ax, bx), (ay, by) = bounds
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


# ----------------------------
# PDE: Poisson κ = 1, Robin with manufactured solution
# ----------------------------
@struct.dataclass
class Poisson2DRobinManufactured:
  """
  Manufactured Poisson problem on Ω = Ω_disk ∪ Ω_rect with κ = 1:

    -Δu = f(x,y)          in Ω,
    u + eps_phys ∂_n u = h(x,y) on ∂Ω_phys,

  where

    u(x,y) = cos(π x) cos(π y),
    f(x,y) = 2 π^2 cos(π x) cos(π y),
    h(x,y) = u + eps_phys ∂_n u.

  In the weak formulation we use:

    a(u,v) = ∫_Ω ∇u·∇v + eps_phys^{-1} ∫_{∂Ω_phys} u v,
    L(v)   = ∫_Ω f v + eps_phys^{-1} ∫_{∂Ω_phys} h v.

  Physical boundary is detected via quad.boundary_mask_global; interface
  contributions are handled separately by DDPDE/eps_interface.
  """

  eps: float = 1e-2

  # exact solution and its gradient
  def exact_solution(self):
    def u_exact(X: jax.Array) -> jax.Array:
      X = jnp.asarray(X)
      x = X[:, 0:1]
      y = X[:, 1:2]
      val = jnp.cos(jnp.pi * x) * jnp.cos(jnp.pi * y)
      return val
    return u_exact

  def grad_exact(self):
    def grad_u(X: jax.Array) -> jax.Array:
      X = jnp.asarray(X)
      x = X[:, 0:1]
      y = X[:, 1:2]
      ux = -jnp.pi * jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y)
      uy = -jnp.pi * jnp.cos(jnp.pi * x) * jnp.sin(jnp.pi * y)
      return jnp.concatenate([ux, uy], axis=1)
    return grad_u

  def source(self):
    # f(x,y) = 2 π^2 cos(π x) cos(π y)
    def f(X: jax.Array) -> jax.Array:
      X = jnp.asarray(X)
      x = X[:, 0:1]
      y = X[:, 1:2]
      val = 2.0 * (jnp.pi ** 2) * jnp.cos(jnp.pi * x) * jnp.cos(jnp.pi * y)
      return val
    return f

  def linear_operator(self):
    u_exact = self.exact_solution()
    grad_u = self.grad_exact()
    f = self.source()
    inv_eps = 1.0 / self.eps

    def L(v: FunctionState, quad) -> jax.Array:
      # interior contribution
      fvals = f(quad.interior_x)
      Li = jnp.einsum("n,ni->i", quad.interior_w, (fvals * v.interior)[:, :])

      # physical boundary Robin RHS: eps^{-1} ∫ h v, with h = u + eps ∂_n u
      gamma_b = quad.boundary_w
      mask_g = getattr(quad, "boundary_mask_global", None)
      if mask_g is not None:
        gamma_b = gamma_b * mask_g.astype(gamma_b.dtype)

      Xb = quad.boundary_x
      ub = u_exact(Xb)              # (Nb,1)
      gradb = grad_u(Xb)            # (Nb,2)
      n = quad.boundary_normal      # (Nb,2), outward w.r.t subdomain
      n_dot_grad = jnp.sum(gradb * n, axis=1, keepdims=True)

      hvals = ub + self.eps * n_dot_grad  # (Nb,1)

      Lb = jnp.einsum("n,ni->i", inv_eps * gamma_b, (hvals * v.boundary)[:, :])

      return (Li + Lb).reshape(1, -1)

    return L

  def bilinear_form(self):
    inv_eps = 1.0 / self.eps

    def a(u: FunctionState, v: FunctionState, quad) -> jax.Array:
      # ∫_Ω ∇u·∇v
      a_grad = jnp.einsum(
        "nui,nvi,n->uv",
        u.grad_interior, v.grad_interior, quad.interior_w
      )

      # eps^{-1} ∫_{∂Ω_phys} u v
      gamma_b = quad.boundary_w
      mask_g = getattr(quad, "boundary_mask_global", None)
      if mask_g is not None:
        gamma_b = gamma_b * mask_g.astype(gamma_b.dtype)

      a_bnd = jnp.einsum(
        "an,am,a->nm",
        u.boundary, v.boundary, inv_eps * gamma_b
      )

      return a_grad + a_bnd

    return a

  def energy_norm(self):
    inv_eps = 1.0 / self.eps

    def norm(v: FunctionState, quad) -> jax.Array:
      # interior: ∫_Ω |∇v|^2
      grad_sq = jnp.sum(v.grad_interior ** 2, axis=2)  # (N_interior, n_vec)
      a1 = jnp.einsum("n,ni->i", quad.interior_w, grad_sq)

      # boundary: eps^{-1} ∫_{∂Ω_phys} |v|^2
      b_sq = v.boundary ** 2                             # (N_bnd, n_vec)
      gamma_b = quad.boundary_w                         # (N_bnd,)
      mask_g = getattr(quad, "boundary_mask_global", None)
      if mask_g is not None:
        gamma_b = gamma_b * mask_g.astype(gamma_b.dtype)

      a2 = jnp.einsum("n,ni->i", gamma_b, b_sq)

      en2 = a1 + inv_eps * a2
      en2 = jnp.maximum(en2, jnp.array(0.0, en2.dtype))
      return jnp.sqrt(en2)  # shape (n_vec,)

    return norm
# ===========================
# ======== MAIN ============
# ===========================
Q_disk, Q_rect = dd_overlapping_disk_rectangle_quadratures(
  R=R,
  rect_bounds=rect_bounds,
  n_r=n_r,
  n_theta=n_theta,
  nx=nx_rect,
  ny=ny_rect,
  n_edge=n_edge_rect,
)

print(f"Disk: n_int={Q_disk.n_interior}, n_bnd={Q_disk.n_boundary}")
print(f"Rect: n_int={Q_rect.n_interior}, n_bnd={Q_rect.n_boundary}")

pde_base = Poisson2DRobinManufactured(eps=eps_phys)
u_exact_fn = pde_base.exact_solution()

# Initial guesses
z = lambda X: jnp.zeros((X.shape[0], 1))
gradz = lambda X: jnp.zeros((X.shape[0], 2))

u0_state = FunctionState.from_function(z, Q_disk, gradz)
u1_state = FunctionState.from_function(z, Q_rect, gradz)

g0 = z  # trace from rectangle to disk
g1 = z  # trace from disk to rectangle

u0_fn = z
u1_fn = z

start = time.perf_counter()

for k in range(max_sweeps):
  print(f"===== Schwarz sweep {k + 1} / {max_sweeps} =====")

  # ---------- Ω0: disk ----------
  pde0 = DDPDE(base=pde_base, eps_interface=eps_interface, trace_fns=(g0,))
  out0 = GalerkinNN(pde0, Q_disk).solve(
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
  u0_state = u0_state_out
  u0_coeff_vec = as_coeff_vector(u0_coeff)
  u0_fn = make_u_fn(sigma_list0, u0_coeff_vec, basis_coeff_list0)
  grad_u0_fn = make_grad2d(u0_fn)

  # interface trace from disk -> rectangle
  def g1_new(X, grad_fn=grad_u0_fn):
    X = jnp.asarray(X)
    n = radial_normal(X)    # outward normal of disk
    grad = grad_fn(X)
    n_dot_grad = jnp.sum(grad * n, axis=1, keepdims=True)
    return u0_fn(X) + eps_interface * (kappa * n_dot_grad)

  g1 = relax_fn(g1_new, g1, omega)

  # ---------- Ω1: rectangle ----------
  pde1 = DDPDE(base=pde_base, eps_interface=eps_interface, trace_fns=(g1,))
  out1 = GalerkinNN(pde1, Q_rect).solve(
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
  u1_state = u1_state_out
  u1_coeff_vec = as_coeff_vector(u1_coeff)
  u1_fn = make_u_fn(sigma_list1, u1_coeff_vec, basis_coeff_list1)
  grad_u1_fn = make_grad2d(u1_fn)

  # interface trace from rectangle -> disk
  def g0_new(X, grad_fn=grad_u1_fn, bounds=rect_bounds):
    X = jnp.asarray(X)
    n = rectangle_normal(X, bounds)  # outward normal of rectangle
    grad = grad_fn(X)
    n_dot_grad = jnp.sum(grad * n, axis=1, keepdims=True)
    return u1_fn(X) + eps_interface * (kappa * n_dot_grad)

  g0 = relax_fn(g0_new, g0, omega)

elapsed = time.perf_counter() - start
print(f"Total elapsed time: {elapsed:.3f} s")


# -----------------------
# Stitch global solution
# -----------------------
w_disk_fn, w_rect_fn = build_pou_weights_disk_rect(Q_disk, Q_rect)

def u_global_fn(X):
  X = jnp.asarray(X).reshape(-1, 2)
  return w_disk_fn(X) * u0_fn(X) + w_rect_fn(X) * u1_fn(X)

# %%
# -----------------------
# Error & visualization
# -----------------------
# simple uniform grid over a bounding box, restricted to the union domain
xmin = min(-R, rect_bounds[0][0]) - 0.05
xmax = max(R,  rect_bounds[0][1]) + 0.05
ymin = min(-R, rect_bounds[1][0]) - 0.05
ymax = max(R,  rect_bounds[1][1]) + 0.05

Nx = 200
Ny = 200
xg = jnp.linspace(xmin, xmax, Nx)
yg = jnp.linspace(ymin, ymax, Ny)
Xg, Yg = jnp.meshgrid(xg, yg, indexing="ij")
X_flat = jnp.stack([Xg.ravel(), Yg.ravel()], axis=1)

# membership for the union domain
r2 = jnp.sum(X_flat**2, axis=1)
in_disk = r2 <= (R**2 + 1e-8)

x = X_flat[:, 0]
y = X_flat[:, 1]
(ax, bx), (ay, by) = rect_bounds
in_rect = (x >= ax) & (x <= bx) & (y >= ay) & (y <= by)

in_union = in_disk | in_rect

# evaluate solution ONLY on union points
X_union = X_flat[in_union]
u_num_union = jnp.squeeze(u_global_fn(X_union))
u_ex_union  = jnp.squeeze(u_exact_fn(X_union))
err_union   = u_num_union - u_ex_union

# norms on the union
Linf = float(jnp.max(jnp.abs(err_union)))
L2   = float(jnp.sqrt(jnp.mean(err_union**2)))  # RMS on union sample
print(f"[analytic] L∞(Ω) ≈ {Linf:.3e},  RMS(Ω) ≈ {L2:.3e}")

# ---- build NaN-masked grids for plotting ----
in_union_np = np.array(in_union)

U_num = np.full((Nx * Ny,), np.nan)
U_ex  = np.full((Nx * Ny,), np.nan)
U_err = np.full((Nx * Ny,), np.nan)

U_num[in_union_np] = np.array(u_num_union).ravel()
U_ex[in_union_np]  = np.array(u_ex_union).ravel()
U_err[in_union_np] = np.abs(np.array(err_union)).ravel()

U_num = U_num.reshape(Nx, Ny)
U_ex  = U_ex.reshape(Nx, Ny)
U_err = U_err.reshape(Nx, Ny)

# ---- plots: use pcolormesh with NaNs outside Ω ----
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.pcolormesh(np.array(xg), np.array(yg), U_num.T, shading="auto")
plt.gca().set_aspect("equal", adjustable="box")
plt.colorbar(label="u_h")
plt.title("Numerical u_h")

plt.subplot(1, 3, 2)
plt.pcolormesh(np.array(xg), np.array(yg), U_ex.T, shading="auto")
plt.gca().set_aspect("equal", adjustable="box")
plt.colorbar(label="u_exact")
plt.title("Exact u")

plt.subplot(1, 3, 3)
plt.pcolormesh(np.array(xg), np.array(yg), U_err.T, shading="auto")
plt.gca().set_aspect("equal", adjustable="box")
plt.colorbar(label="|u_h - u_exact|")
plt.title("Absolute error")

plt.tight_layout()
plt.show()

# %%
# -----------------------
# Alternative plot: sample on union of quadrature points (deduped)
# -----------------------
def _unique_union_points(Q0, Q1, decimals: int = 6):
  """Stack interior nodes from both subdomains and drop near-duplicates."""
  X0 = np.asarray(Q0.interior_x)
  X1 = np.asarray(Q1.interior_x)
  X_all = np.vstack([X0, X1])
  key = np.round(X_all, decimals=decimals)
  _, idx = np.unique(key, axis=0, return_index=True)
  idx_sorted = np.sort(idx)
  return X_all[idx_sorted]


X_union_q = _unique_union_points(Q_disk, Q_rect, decimals=6)
u_num_q = np.squeeze(np.asarray(u_global_fn(X_union_q)))
u_ex_q = np.squeeze(np.asarray(u_exact_fn(X_union_q)))
err_q = u_num_q - u_ex_q

# Filled visualization on the union (triangulated)
import matplotlib.tri as mtri

def _in_union_bool(X):
  x = X[:, 0]
  y = X[:, 1]
  r2 = x**2 + y**2
  in_disk = r2 <= (R**2 + 1e-10)
  (ax, bx), (ay, by) = rect_bounds
  in_rect = (x >= ax) & (x <= bx) & (y >= ay) & (y <= by)
  return in_disk | in_rect



tri = mtri.Triangulation(X_union_q[:, 0], X_union_q[:, 1])
tri_pts = X_union_q[tri.triangles]        # (ntri, 3, 2)
centroids = np.mean(tri_pts, axis=1)      # (ntri, 2)
mask = ~_in_union_bool(centroids)         # (ntri,)
tri.set_mask(mask)

fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))

tpc = axs[0].tripcolor(tri, u_num_q, shading="gouraud", cmap="viridis")
axs[0].set_aspect("equal", adjustable="box")
axs[0].set_title("u_h (triangulated PoU blend)")
fig.colorbar(tpc, ax=axs[0], shrink=0.85, label="u_h")

tpc_err = axs[1].tripcolor(tri, np.abs(err_q), shading="gouraud", cmap="magma_r")
axs[1].set_aspect("equal", adjustable="box")
axs[1].set_title("|u_h - u_exact|")
fig.colorbar(tpc_err, ax=axs[1], shrink=0.85, label="abs error")

plt.tight_layout()
plt.show()
# %%
fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))

tpc = axs[0].tripcolor(tri, u_num_q, shading="gouraud", cmap="viridis")
axs[0].set_aspect("equal", adjustable="box")
axs[0].set_title("u_h (triangulated PoU blend)")
fig.colorbar(tpc, ax=axs[0], shrink=0.85, label="u_h")

tpc_err = axs[1].tripcolor(tri, np.abs(err_q), shading="gouraud", cmap="magma_r")
axs[1].set_aspect("equal", adjustable="box")
axs[1].set_title("|u_h - u_exact|")
fig.colorbar(tpc_err, ax=axs[1], shrink=0.85, label="abs error")

plt.tight_layout()
for fmt, dpi in (("png", 300), ("pdf", None)):
  savepath = images_path / f"{EXPERIMENT}_solution.{fmt}"
  save_kwargs = {"bbox_inches": "tight", "transparent": True}
  if dpi is not None:
    save_kwargs["dpi"] = dpi
  fig.savefig(savepath, **save_kwargs)
plt.close(fig)
