# steady_heat2D_robin_disk_piecewiseK.py
# %%
import jax
import jax.numpy as jnp
import optax
from typing import Callable
from flax import struct

from galerkinnn import FunctionState, PDE, Quadrature, GalerkinNN
from galerkinnn.quadratures import disk_quadrature

# -------------------------
# Hyper-parameters
# -------------------------
seed = 43
max_bases = 5
max_epoch_basis = 50
tol_solution = 1e-7
tol_basis = 1e-7

N = 5
r = 2
A = 5e-3
rho = 1.1

@struct.dataclass
class SteadyHeat2DRobin(PDE):
  k_fn:      Callable = struct.field(pytree_node=False)
  f_fn:      Callable = struct.field(pytree_node=False)
  h_bdry_fn: Callable = struct.field(pytree_node=False)
  g_bdry_fn: Callable = struct.field(pytree_node=False)

  def source(self): return self.f_fn

  def linear_operator(self):
    f_fn = self.f_fn
    g_bdry_fn = self.g_bdry_fn
    def L(v, quad) -> jnp.ndarray:
      fvals = f_fn(quad.interior_x).reshape(-1)
      Lin = jnp.sum(
        (fvals[:, None] * v.interior) * quad.interior_w[:, None],
        axis=0,
        keepdims=True
      )
      gvals = g_bdry_fn(quad.boundary_x).reshape(-1)
      Lbd = jnp.sum(
        (gvals[:, None] * v.boundary) * quad.boundary_w[:, None],
        axis=0,
        keepdims=True
      )
      return Lin + Lbd
    return L

  def bilinear_form(self):
    k_fn = self.k_fn
    h_bdry_fn = self.h_bdry_fn
    def a(u, v, quad) -> jnp.ndarray:
      kvals = k_fn(quad.interior_x).reshape(-1)
      a1 = jnp.einsum(
        "nui,nvi,n->uv",
        u.grad_interior,
        v.grad_interior,
        kvals * quad.interior_w
      )
      hvals = h_bdry_fn(quad.boundary_x).reshape(-1)
      a2 = jnp.einsum(
        "an,am,a->nm",
        u.boundary,
        v.boundary,
        hvals * quad.boundary_w
      )
      return a1 + a2
    return a

  def energy_norm(self):
    k_fn = self.k_fn
    h_bdry_fn = self.h_bdry_fn
    def norm(v, quad) -> jax.Array:
      kvals = k_fn(quad.interior_x).reshape(-1)
      grad_sq = jnp.sum(v.grad_interior**2, axis=2)
      a1 = jnp.einsum("n,ni->i", kvals * quad.interior_w, grad_sq)
      hvals = h_bdry_fn(quad.boundary_x).reshape(-1)
      b_sq = v.boundary**2
      a2 = jnp.einsum("n,ni->i", hvals * quad.boundary_w, b_sq)
      en2 = jnp.maximum(a1 + a2, jnp.array(0., v.interior.dtype))
      return jnp.sqrt(en2)
    return norm

def net_fn(X, params, activation):
  X = jnp.dot(X, params["W"]) + params["b"]
  return activation(X)

def activations_fn(i: int):
  s = 1.5 * i
  return lambda x: jnp.tanh(s * x)

WIDTH_CAP = 512
network_widths_fn = lambda i: min(WIDTH_CAP, N * r**(i-1))

# network_widths_fn = lambda i: N * r ** (i - 1)
learning_rates_fn = lambda i: A * rho ** (-(i - 1))

# geometry
R = 1.0
nr, nt = 220, 256         # a bit denser helps across jumps
quad = disk_quadrature(radius=R, n_r=nr, n_theta=nt)

# -------------------------
# Piecewise-constant k in three annuli
# r < r1         : k = k1 (low)
# r1 <= r < r2   : k = k2 (mid)
# r2 <= r <= R   : k = k3 (high)
# -------------------------
r1, r2 = 0.35, 0.72
k1, k2, k3 = 0.4, 1.0, 4.0

# def k_piecewise(X):
#   x = X[:, 0:1]
#   y = X[:, 1:2]            # ← FIX
#   r = jnp.sqrt(x**2 + y**2)
#   k = jnp.where(r < r1, k1, jnp.where(r < r2, k2, k3))
#   return k                 # (N,1)

def smooth_H(t, eps=0.01):  # 0.005–0.02 works well
  return 0.5 * (1.0 + jnp.tanh(t/eps))

def k_piecewise(X, eps=0.01):
  # Smoothed
  x, y = X[:,0:1], X[:,1:2]
  r = jnp.sqrt(x**2 + y**2)
  H1 = smooth_H(r1 - r, eps)   # inner
  H3 = smooth_H(r - r2, eps)   # outer
  H2 = 1.0 - H1 - H3           # middle
  return k1*H1 + k2*H2 + k3*H3


def f_bump(X):
  x = X[:, 0:1]
  y = X[:, 1:2]
  src = 25.0 * jnp.exp(-((x - 0.0)**2 + (y - 0.65)**2) / 0.01)
  return 1.0 + src         # (N,1)


H = 8.0
Tinf = 0.0
h_bdry_fn = lambda Xb: jnp.ones((Xb.shape[0], 1)) * H
g_bdry_fn = lambda Xb: jnp.ones((Xb.shape[0], 1)) * (H * Tinf)

pde = SteadyHeat2DRobin(
  k_fn=lambda X: k_piecewise(X),
  f_fn=f_bump,
  h_bdry_fn=h_bdry_fn,
  g_bdry_fn=g_bdry_fn
)

u0_fn = lambda X: jnp.zeros((X.shape[0], 1))
u0_grad = lambda X: jnp.zeros_like(X)
u0 = FunctionState.from_function(u0_fn, quad=quad, grad_func=u0_grad)

# solve
import time
start = time.perf_counter()
solver = GalerkinNN(pde, quad)
u, u_coeff, eta_errors, basis_list, basis_params_list, basis_coeff_list, sigma_net_fn_list = solver.solve(
  seed=seed,
  u0=u0,
  net_fn=net_fn,
  activations_fn=activations_fn,
  network_widths_fn=network_widths_fn,
  learning_rates_fn=learning_rates_fn,
  max_bases=max_bases,
  max_epoch_basis=max_epoch_basis,
  tol_solution=tol_solution,
  tol_basis=tol_basis,
)
print(f"Elapsed: {time.perf_counter()-start:.3f}s")

# %%
# interpolated visualization (temperature and k)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator

x = quad.interior_x[:, 0]; y = quad.interior_x[:, 1]
tri = Triangulation(x, y)

u_nn = u.interior[:, 0]
Iu = LinearTriInterpolator(tri, np.array(u_nn))

k_vals = np.array(k_piecewise(quad.interior_x)[:, 0])
Ik = LinearTriInterpolator(tri, k_vals)

n = 480
xg = np.linspace(x.min(), x.max(), n)
yg = np.linspace(y.min(), y.max(), n)
Xg, Yg = np.meshgrid(xg, yg, indexing='xy')
mask = (Xg**2 + Yg**2) > (R*R)

Zu = np.array(Iu(Xg, Yg))
Zk = np.array(Ik(Xg, Yg))
Zu = np.ma.array(Zu, mask=mask)
Zk = np.ma.array(Zk, mask=mask)

fig, ax = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
im0 = ax[0].pcolormesh(Xg, Yg, Zu, shading='auto', cmap='viridis')
ax[0].set_title("NN temperature (interpolated)"); ax[0].set_aspect('equal')
plt.colorbar(im0, ax=ax[0])

im1 = ax[1].pcolormesh(Xg, Yg, Zk, shading='auto', cmap='magma')
ax[1].set_title("Piecewise k(x,y) (interpolated)"); ax[1].set_aspect('equal')
plt.colorbar(im1, ax=ax[1])
fig.savefig("images/steady_heat2d_robin_disk_piecewiseK.png", dpi=160)


# %%
# =========================
# Robust diagnostics (adaptive bands)
# =========================
import jax
import jax.numpy as jnp
from galerkinnn import FunctionState
from galerkinnn.solver import spd_solve

def _grad_single(G):
  # Accept (N,1,dim) or (N,dim) and return (N,dim)
  return G[:, 0, :] if (G.ndim == 3 and G.shape[1] == 1) else G

def _adaptive_band_mask(r, R, side: str, min_pts: int):
  """
  Pick the closest 'min_pts' points to the interface at r=R on the chosen side.
  side = 'inside' (r<R) or 'outside' (r>R).
  Returns a boolean mask of length N.
  """
  if side == 'inside':
    sel = r < R
  elif side == 'outside':
    sel = r > R
  else:
    raise ValueError("side must be 'inside' or 'outside'")
  # distances to interface
  d = jnp.abs(r - R)
  d_sel = jnp.where(sel, d, jnp.inf)
  # threshold so that at least min_pts points are included
  k = jnp.minimum(min_pts, d_sel.size)
  # guard if there are too few points on one side (extreme R)
  k = jnp.maximum(k, 1)
  # jnp.partition returns the kth order statistic; we use k-1 for 0-based
  thr = jnp.sort(d_sel)[k - 1]
  mask = sel & (d <= thr)
  return mask

def check_interface_flux_adaptive(u, quad, k_fn, radii, min_pts_side=256):
  """
  Flux continuity across interfaces r=R: compares weighted averages of k ∂_n u
  from the inside and outside using adaptive thin shells with at least min_pts_side points.
  """
  X = quad.interior_x
  w = quad.interior_w
  r = jnp.sqrt(jnp.sum(X**2, axis=1))
  n = X / (r[:, None] + 1e-12)                  # outward radial normals
  G = _grad_single(u.grad_interior)             # (N,2)
  kvals = k_fn(X)[:, 0]                         # (N,)
  qn = kvals * jnp.sum(G * n, axis=1)           # (N,)

  for R in radii:
    m_in  = _adaptive_band_mask(r, R, 'inside',  min_pts_side)
    m_out = _adaptive_band_mask(r, R, 'outside', min_pts_side)

    # weighted averages
    w_in  = jnp.sum(w[m_in])
    w_out = jnp.sum(w[m_out])
    avg_in  = jnp.sum(qn[m_in]  * w[m_in])  / (w_in  + 1e-30)
    avg_out = jnp.sum(qn[m_out] * w[m_out]) / (w_out + 1e-30)
    rel = jnp.abs(avg_in - avg_out) / (jnp.abs(avg_in) + jnp.abs(avg_out) + 1e-30)

    print(f"r={R:.3f}:  <k∂_n u>_in={float(avg_in):+.4e}  "
          f"<k∂_n u>_out={float(avg_out):+.4e}  "
          f"rel={float(rel):.2e}  "
          f"(pts_in={int(m_in.sum())}, pts_out={int(m_out.sum())})")

def check_robin_residual_weighted(u, quad, k_fn, h_bdry_fn, g_bdry_fn):
  """
  Weighted boundary residual for Robin: k ∂_n u + h u - g.
  Prints max|·|, L2, and relative L2.
  """
  Xb  = quad.boundary_x
  nb  = quad.boundary_normal
  wb  = quad.boundary_w
  ub  = u.boundary[:, 0]
  hb  = h_bdry_fn(Xb)[:, 0]
  gb  = g_bdry_fn(Xb)[:, 0]
  kb  = k_fn(Xb)[:, 0]
  Gb  = _grad_single(u.grad_boundary)
  gnb = jnp.sum(Gb * nb, axis=1)
  res = kb * gnb + hb * ub - gb

  L2  = jnp.sqrt(jnp.sum(wb * res**2))
  den = jnp.sqrt(jnp.sum(wb * (hb * ub - gb)**2)) + 1e-30
  rel = L2 / den
  print(f"Robin residual:  max|·| = {float(jnp.max(jnp.abs(res))):.6e},  "
        f"L2 = {float(L2):.6e},  rel = {float(rel):.6e}")

def check_galerkin_residual_strict(pde, quad, basis_list, u_coeff, ridge_used=1e-8):
  """
  Re-assembles K and F like the solver and checks algebraic residuals.
  Also resolves once to compare coefficients.
  """
  bases = FunctionState(
    interior=jnp.concatenate([b.interior for b in basis_list], axis=1),
    boundary=jnp.concatenate([b.boundary for b in basis_list], axis=1),
    grad_interior=jnp.concatenate([b.grad_interior for b in basis_list], axis=1),
    grad_boundary=jnp.concatenate([b.grad_boundary for b in basis_list], axis=1),
  )

  a = pde.bilinear_form()
  L = pde.linear_operator()

  K = a(bases, bases, quad)                      # (m,m)
  F = L(bases, quad).T                           # (m,1)

  r_raw = K @ u_coeff - F
  Ksym  = 0.5 * (K + K.T)
  r_sym = Ksym @ u_coeff - F

  Fnorm   = jnp.linalg.norm(F)
  rel_raw = jnp.linalg.norm(r_raw) / (Fnorm + 1e-30)
  rel_sym = jnp.linalg.norm(r_sym) / (Fnorm + 1e-30)

  s = jnp.linalg.svd(Ksym, compute_uv=False)
  cond = float(s.max() / jnp.maximum(s.min(), 1e-30))

  print(f"K shape={K.shape}, F shape={F.shape}, u_coeff shape={u_coeff.shape}")
  print(f"||r_raw||2  = {float(jnp.linalg.norm(r_raw)):.3e}")
  print(f"||r_sym||2  = {float(jnp.linalg.norm(r_sym)):.3e}")
  print(f"rel_raw     = {float(rel_raw):.3e}")
  print(f"rel_sym     = {float(rel_sym):.3e}")
  print(f"cond(Ksym)  = {cond:.3e}")

  # resolve with same K,F to sanity-check
  u_coeff_ref = spd_solve(K, F, ridge=ridge_used)
  r_ref = K @ u_coeff_ref - F
  rel_ref = jnp.linalg.norm(r_ref) / (Fnorm + 1e-30)
  diff   = jnp.linalg.norm(u_coeff_ref - u_coeff) / (jnp.linalg.norm(u_coeff_ref) + 1e-30)
  print(f"ref rel (solve now) = {float(rel_ref):.3e}")
  print(f"rel ||u_coeff_ref - u_coeff|| = {float(diff):.3e}")

# ---- Run all checks
print("=== DIAGNOSTICS (adaptive) ===")
check_interface_flux_adaptive(u, quad, k_piecewise, radii=[r1, r2], min_pts_side=256)
check_robin_residual_weighted(u, quad, k_piecewise, h_bdry_fn, g_bdry_fn)
check_galerkin_residual_strict(pde, quad, basis_list, u_coeff, ridge_used=1e-8)

# %%
