# steady_heat2D_robin_disk_varK_complex.py
# %%
import jax
import jax.numpy as jnp
import optax
from typing import Callable
from flax import struct

from galerkinnn import FunctionState, PDE, Quadrature, GalerkinNN
from galerkinnn.quadratures import gauss_legendre_disk_quadrature

# -------------------------
# Hyper-parameters
# -------------------------
seed = 42
max_bases = 4
max_epoch_basis = 120
tol_solution = 1e-7
tol_basis = 1e-7

N = 5
r = 2
A = 5e-3
rho = 1.1

# -------------------------
# PDE class
# -------------------------
@struct.dataclass
class SteadyHeat2DRobin(PDE):
  k_fn:      Callable = struct.field(pytree_node=False)  # (N,2)->(N,1), k>=k_min>0
  f_fn:      Callable = struct.field(pytree_node=False)  # (N,2)->(N,1)
  h_bdry_fn: Callable = struct.field(pytree_node=False)  # (Nb,2)->(Nb,1)
  g_bdry_fn: Callable = struct.field(pytree_node=False)  # (Nb,2)->(Nb,1)

  def source(self): return self.f_fn

  def linear_operator(self):
    f_fn = self.f_fn
    g_bdry_fn = self.g_bdry_fn
    def L(v: FunctionState, quad: Quadrature) -> jnp.ndarray:
      fvals = f_fn(quad.interior_x).reshape(-1)
      Lin = jnp.sum((fvals[:, None] * v.interior) * quad.interior_w[:, None],
                    axis=0, keepdims=True)
      gvals = g_bdry_fn(quad.boundary_x).reshape(-1)
      Lbd = jnp.sum((gvals[:, None] * v.boundary) * quad.boundary_w[:, None],
                    axis=0, keepdims=True)
      return Lin + Lbd
    return L

  def bilinear_form(self):
    k_fn = self.k_fn
    h_bdry_fn = self.h_bdry_fn
    def a(u: FunctionState, v: FunctionState, quad: Quadrature) -> jnp.ndarray:
      kvals = k_fn(quad.interior_x).reshape(-1)
      a1 = jnp.einsum("nui,nvi,n->uv",
                      u.grad_interior, v.grad_interior, kvals * quad.interior_w)
      hvals = h_bdry_fn(quad.boundary_x).reshape(-1)
      a2 = jnp.einsum("an,am,a->nm",
                      u.boundary, v.boundary, hvals * quad.boundary_w)
      return a1 + a2
    return a

  def energy_norm(self):
    k_fn = self.k_fn
    h_bdry_fn = self.h_bdry_fn
    def norm(v: FunctionState, quad: Quadrature) -> jax.Array:
      kvals = k_fn(quad.interior_x).reshape(-1)
      grad_sq = jnp.sum(v.grad_interior**2, axis=2)
      a1 = jnp.einsum("n,ni->i", kvals * quad.interior_w, grad_sq)
      hvals = h_bdry_fn(quad.boundary_x).reshape(-1)
      b_sq = v.boundary**2
      a2 = jnp.einsum("n,ni->i", hvals * quad.boundary_w, b_sq)
      en2 = jnp.maximum(a1 + a2, jnp.array(0., v.interior.dtype))
      return jnp.sqrt(en2)
    return norm

# -------------------------
# Neural network basis
# -------------------------
def net_fn(X, params, activation):
  X = jnp.dot(X, params["W"]) + params["b"]
  return activation(X)

def activations_fn(i): return lambda x: jnp.tanh(i * x)
network_widths_fn = lambda i: N * r ** (i - 1)
learning_rates_fn = lambda i: A * rho ** (-(i - 1))

# -------------------------
# Geometry & quadrature
# -------------------------
R = 1.0
nr, nt = 128, 128
quad = gauss_legendre_disk_quadrature(nr=nr, nt=nt, R=R)

# -------------------------
# Variable k(x,y) with angular + radial modulation and two Gaussian bumps.
# Guaranteed positive.
# -------------------------
def k_complex(X):
  x = X[:, 0:1]
  y = X[:, 1:2]            # ← FIX: 1:2 (not 1:1)
  r2 = x**2 + y**2
  theta = jnp.arctan2(y, x)
  ang = 0.8 * (1.0 + 0.9 * jnp.cos(6.0 * theta)) * (0.2 + r2)
  g1 = 2.5 * jnp.exp(-((x - 0.45)**2 + (y - 0.15)**2) / 0.02)
  g2 = 1.5 * jnp.exp(-((x + 0.4)**2 + (y + 0.4)**2) / 0.01)
  return 1.0 + ang + g1 + g2  # (N,1)

def f_local(X):
  x = X[:, 0:1]
  y = X[:, 1:2]            # ← FIX
  hot = 10.0 * jnp.exp(-((x - 0.2)**2 + (y + 0.35)**2) / 0.02)
  return 2.0 + hot         # (N,1)


H = 5.0
Tinf = 0.0
h_bdry_fn = lambda Xb: jnp.ones((Xb.shape[0], 1)) * H
g_bdry_fn = lambda Xb: jnp.ones((Xb.shape[0], 1)) * (H * Tinf)

pde = SteadyHeat2DRobin(
  k_fn=k_complex,
  f_fn=f_local,
  h_bdry_fn=h_bdry_fn,
  g_bdry_fn=g_bdry_fn
)

# initial state
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

# interpolated plots
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator

x = quad.interior_x[:, 0]; y = quad.interior_x[:, 1]
tri = Triangulation(x, y)

u_nn = u.interior[:, 0]
I_nn = LinearTriInterpolator(tri, np.array(u_nn))

n = 400
xg = np.linspace(x.min(), x.max(), n)
yg = np.linspace(y.min(), y.max(), n)
Xg, Yg = np.meshgrid(xg, yg, indexing='xy')
Rmask = (Xg**2 + Yg**2) > (R*R)
Z = np.array(I_nn(Xg, Yg))
Z = np.ma.array(Z, mask=Rmask)

fig, ax = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
im0 = ax[0].pcolormesh(Xg, Yg, Z, shading='auto', cmap='viridis')
ax[0].set_title("NN temperature (interpolated)"); ax[0].set_aspect('equal')
plt.colorbar(im0, ax=ax[0])

# show k(x,y) itself
k_vals = np.array(k_complex(quad.interior_x)[:, 0])
Ik = LinearTriInterpolator(tri, k_vals)
Zk = np.array(Ik(Xg, Yg))
Zk = np.ma.array(Zk, mask=Rmask)
im1 = ax[1].pcolormesh(Xg, Yg, Zk, shading='auto', cmap='magma')
ax[1].set_title("Conductivity k(x,y) (interpolated)"); ax[1].set_aspect('equal')
plt.colorbar(im1, ax=ax[1])
fig.savefig("images/steady_heat2d_robin_disk_varK_complex.png", dpi=160)
