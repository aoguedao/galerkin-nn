# %%
import jax
import jax.numpy as jnp
import optax

from typing import Callable

from galerkinnn import FunctionState, PDE, Quadrature, GalerkinNN
from galerkinnn.quadratures import gauss_legendre_disk_quadrature

# Hyper-parameters
seed = 42
max_bases = 6
max_epoch_basis = 100
tol_solution = 1e-7
tol_basis = 1e-7

# n_train = 128
n_val = 1024
N = 5  # Init Neurons
r = 2  # Neurons Growth
A = 5e-3  # Init Learning Rate
rho = 1.1  # Learning Rate Growth


@jax.tree_util.register_dataclass
class PoissonMembraneDisplacementRobinBC(PDE):
  """
  2D Poisson (membrane displacement) on the unit disk with Robin BC:
    -Δu = f  in Ω ⊂ R^2
    u + eps * ∂_n u = 0 on ∂Ω
  Bilinear form:
    a(u,v) = (∇u, ∇v)_Ω + eps^{-1} (u, v)_{∂Ω}
  Linear functional:
    L(v)   = (f, v)_Ω
  Energy norm:
    ||v||_a^2 = ||∇v||_{L^2(Ω)}^2 + eps^{-1} ||v||_{L^2(∂Ω)}^2
  """
  def __init__(self, eps: float = 1e-4, f_const: float = 2.0):
    self.eps = eps
    self._f_const = f_const

  def source(self) -> Callable[[jax.Array], jax.Array]:
    """
    Return f(x,y). Default: constant f ≡ f_const (shape (N,1)).
    """
    c = jnp.array(self._f_const, dtype=jnp.float32)
    def f(X: jax.Array) -> jax.Array:
      # X: (N,2) -> (N,1)
      N = X.shape[0]
      return jnp.full((N, 1), c, dtype=X.dtype)
    return f

  def linear_operator(self) -> Callable:
    f = self.source()
    def L(v: FunctionState, quad) -> jnp.ndarray:
      # v.interior: (NΩ, n_v), f(X): (NΩ,1)
      fvals = f(quad.interior_x)                      # (NΩ,1)
      integrand = fvals * v.interior                  # (NΩ, n_v) via broadcast
      # integrate_interior returns (1, n_v) in your Quadrature
      return quad.integrate_interior(integrand)       # (1, n_v)
    return L

  def bilinear_form(self) -> Callable:
    eps = self.eps
    def a(u: FunctionState, v: FunctionState, quad) -> jnp.ndarray:
      # u.grad_interior, v.grad_interior: (NΩ, n_*, dim=2)
      # interior term: sum_{n} w_n * <∇u, ∇v> -> (n_u, n_v)
      a1 = jnp.einsum(
        "nui,nvi,n->uv",
        u.grad_interior,
        v.grad_interior,
        quad.interior_w
      )
      # boundary term: sum_{a} w_a * u_bnd * v_bnd -> (n_u, n_v)
      a2 = jnp.einsum(
        "an,am,a->nm",
        u.boundary,
        v.boundary,
        quad.boundary_w
      )
      return a1 + (1.0 / eps) * a2                    # (n_u, n_v)
    return a

  def energy_norm(self) -> Callable:
    eps = self.eps
    def norm(v: FunctionState, quad) -> jax.Array:
      # ∥v∥_a^2 = ∑ w_i |∇v|^2 + eps^{-1} ∑ w_a v^2
      # grad term
      grad_sq = jnp.sum(v.grad_interior ** 2, axis=2)         # (NΩ, n_v)
      a1 = jnp.einsum("n,ni->i", quad.interior_w, grad_sq)    # (n_v,)
      # boundary term
      b_sq = v.boundary ** 2                                  # (N∂, n_v)
      a2 = jnp.einsum("n,ni->i", quad.boundary_w, b_sq)       # (n_v,)
      en2 = a1 + (1.0 / eps) * a2
      en2 = jnp.maximum(en2, jnp.array(0., en2.dtype))
      return jnp.sqrt(en2)                                     # (n_v,)
    return norm


# Neural Network
def net_fn(
  X: jax.Array,
  params: optax.Params,
  activation: Callable[[jax.Array], jax.Array]
) -> jax.Array:
  X = jnp.dot(X, params["W"]) + params["b"]
  X = activation(X)
  return X

def activations_fn(i):
  scale_fn = lambda i: i
  scale_i = scale_fn(i)
  def activation(x):
    return jnp.tanh(scale_i * x)
  return activation

network_widths_fn = lambda i: N * r ** (i - 1)
learning_rates_fn = lambda i: A * rho ** (-(i - 1))


# Galerkin Neural Network Solver
R = 1.0
nr, nt = 128, 128  # radius, angle
quad = gauss_legendre_disk_quadrature(nr=nr, nt=nt, R=R)
eps = 1e-4
pde = PoissonMembraneDisplacementRobinBC(eps=eps)
u0_fn = lambda X: jnp.zeros(shape=(X.shape[0], 1))
u0_grad = lambda X: jnp.zeros_like(X)
u0 = FunctionState.from_function(func=u0_fn, quad=quad, grad_func=u0_grad)

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

end = time.perf_counter()
elapsed = end - start
print(f"Elapsed time: {elapsed:.6f} seconds")

# %%
import matplotlib.pyplot as plt

def u_sol(X: jax.Array, eps: float):
  r2 = jnp.sum(X**2, axis=1, keepdims=True)
  return -0.5 * r2 + eps + 0.5

u_actual = u_sol(quad.interior_x, eps)
u_pred = u.interior

x, y = quad.interior_x[:,0], quad.interior_x[:,1]
fig, ax = plt.subplots(1,3, figsize=(15,5))

sc0 = ax[0].scatter(x, y, c=u_actual[:,0], cmap="viridis")
ax[0].set_title("Exact solution")
plt.colorbar(sc0, ax=ax[0])

sc1 = ax[1].scatter(x, y, c=u_pred[:,0], cmap="viridis")
ax[1].set_title("NN solution")
plt.colorbar(sc1, ax=ax[1])

sc2 = ax[2].scatter(x, y, c=(u_pred - u_actual)[:,0], cmap="RdBu")
ax[2].set_title("Error (pred - exact)")
plt.colorbar(sc2, ax=ax[2])

for a in ax: a.set_aspect("equal")
fig.savefig("poisson2d_scatter.png")

# %%
import numpy as np
from matplotlib.tri import Triangulation

x, y = quad.interior_x[:,0], quad.interior_x[:,1]
tri = Triangulation(x, y)

fig, ax = plt.subplots()
tpc = ax.tripcolor(tri, u_pred[:,0] - u_actual[:,0], shading="gouraud", cmap="RdBu")
ax.set_aspect("equal")
plt.colorbar(tpc, ax=ax)
ax.set_title("Error field")
fig.savefig("poisson2d_error.png")

# %%
l2_error = jnp.sqrt(jnp.mean((u_pred - u_actual)**2))
print("L2 error:", float(l2_error))
