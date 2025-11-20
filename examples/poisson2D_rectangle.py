# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

from typing import Callable

from galerkinnn.utils import make_u_fn, compare_num_exact_2d
from galerkinnn import FunctionState, PDE, GalerkinNN
from galerkinnn.quadratures import gauss_lobatto_rectangle_quadrature

# Hyper-parameters
seed = 42
max_bases = 10
max_epoch_basis = 50
tol_solution = 1e-7
tol_basis = 1e-7

# n_train = 128
n_val = 1024
N = 5  # Init Neurons
r = 2  # Neurons Growth
A = 5e-3  # Init Learning Rate
rho = 1.1  # Learning Rate Growth


@jax.tree_util.register_dataclass
class PoissonUnitSquareRobinBC(PDE):
  """
  2D Poisson on Ω=(0,1)^2 with global Robin BC and manufactured solution:
    -∇·(κ ∇u) = f          in Ω
     κ ∂_n u + eps^{-1} u = h   on ∂Ω
  Exact solution:
    u(x,y) = sin(πx) sin(πy) + c
  Data:
    f(x,y) = 2 κ π^2 sin(πx) sin(πy)
    h = κ ∂_n u + eps^{-1} u (evaluated per edge; see boundary_data())
  Variational (tests do not vanish on ∂Ω):
    a(u,v) = κ (∇u, ∇v)_Ω + eps^{-1} (u, v)_{∂Ω}      [SPD]
    L(v)   = (f, v)_Ω + (h, v)_{∂Ω}
  Energy norm:
    ||v||_a^2 = κ ||∇v||_{L^2(Ω)}^2 + eps^{-1} ||v||_{L^2(∂Ω)}^2
  Shapes assumed (as in your example):
    u.interior         : (NΩ, n_u)
    u.grad_interior    : (NΩ, n_u, 2)
    u.boundary         : (N∂, n_u)
    quad.interior_x    : (NΩ, 2)
    quad.interior_w    : (NΩ,)
    quad.boundary_x    : (N∂, 2)   # needed for h(x) evaluation
    quad.boundary_w    : (N∂,)
  """
  def __init__(self, kappa: float = 1.0, eps: float = 1e-1, c: float = 0.3, tol_edge: float = 1e-7):
    self.kappa = float(kappa)
    self.eps = float(eps)
    self.c = float(c)
    self.tol_edge = float(tol_edge)

  # ---------- exact solution & data ----------
  def exact_solution(self):
    """Return callable u_exact(X): (N,2)->(N,1)."""
    kappa = self.kappa  # unused but keeps signature pattern
    c = jnp.array(self.c, dtype=jnp.float32)
    def u_exact(X: jax.Array) -> jax.Array:
      x, y = X[:, 0], X[:, 1]
      val = jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y) + c
      return val.reshape(-1, 1)
    return u_exact

  def source(self):
    """Return f(X): (N,2)->(N,1) with f = 2 κ π^2 sin(πx) sin(πy)."""
    kappa = jnp.array(self.kappa, dtype=jnp.float32)
    def f(X: jax.Array) -> jax.Array:
      x, y = X[:, 0], X[:, 1]
      val = 2.0 * kappa * (jnp.pi ** 2) * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)
      return val.reshape(-1, 1)
    return f

  def boundary_data(self):
    """
    Return h(B): (N∂,2)->(N∂,1) for Robin: κ ∂_n u + eps^{-1} u on ∂Ω.
    Uses axis-aligned edge classification with tolerance tol_edge.
      Left/Right (x≈0 or x≈1): h = -κ π sin(π y) + eps^{-1} c
      Bottom/Top (y≈0 or y≈1): h = -κ π sin(π x) + eps^{-1} c
    """
    kappa = jnp.array(self.kappa, dtype=jnp.float32)
    inv_eps_c = jnp.array(self.c / self.eps, dtype=jnp.float32)
    tol = jnp.array(self.tol_edge, dtype=jnp.float32)

    def h(B: jax.Array) -> jax.Array:
      x, y = B[:, 0], B[:, 1]
      left   = x <= tol
      right  = x >= (1.0 - tol)
      bottom = y <= tol
      top    = y >= (1.0 - tol)

      # Vertical edges: normal ±ex → ∂_n u = ±u_x; u_x(0/1,y) = -π sin(π y)
      hy = -kappa * jnp.pi * jnp.sin(jnp.pi * y) + inv_eps_c
      # Horizontal edges: normal ±ey → ∂_n u = ±u_y; u_y(x,0/1) = -π sin(π x)
      hx = -kappa * jnp.pi * jnp.sin(jnp.pi * x) + inv_eps_c
      edge_val = jnp.zeros_like(x)
      edge_val = jnp.where(left | right, hy, edge_val)
      edge_val = jnp.where(bottom | top, hx, edge_val)
      return edge_val.reshape(-1, 1)
    return h

  # ---------- linear functional L ----------
  def linear_operator(self):
    """
    L(v) = (f, v)_Ω + (h, v)_{∂Ω}
    """
    f = self.source()
    h = self.boundary_data()
    def L(v: FunctionState, quad) -> jnp.ndarray:
      # Interior term
      fvals = f(quad.interior_x)              # (NΩ,1)
      Li = jnp.einsum("n,ni->i", quad.interior_w, (fvals * v.interior)[:, :])  # (n_v,)

      # Boundary term
      hvals = h(quad.boundary_x)              # (N∂,1)
      Lb = jnp.einsum("n,ni->i", quad.boundary_w, (hvals * v.boundary)[:, :])  # (n_v,)

      return (Li + Lb).reshape(1, -1)        # (1, n_v) to match your example
    return L

  # ---------- bilinear form a ----------
  def bilinear_form(self):
    """
    a(u,v) = κ (∇u, ∇v)_Ω + eps^{-1} (u, v)_{∂Ω}
    """
    kappa = self.kappa
    inv_eps = 1.0 / self.eps

    def a(u: FunctionState, v: FunctionState, quad) -> jnp.ndarray:
      # Grad term: sum_n w_n * <∇u, ∇v> → (n_u, n_v)
      a_grad = jnp.einsum(
        "nui,nvi,n->uv",
        u.grad_interior, v.grad_interior, quad.interior_w
      )
      # Boundary mass term: sum_a w_a * u_bnd * v_bnd → (n_u, n_v)
      a_bnd = jnp.einsum(
        "an,am,a->nm",
        u.boundary, v.boundary, quad.boundary_w
      )
      return kappa * a_grad + inv_eps * a_bnd
    return a

  # ---------- energy norm ----------
  def energy_norm(self):
    """
    ||v||_a^2 = κ ∫Ω |∇v|^2 + eps^{-1} ∫∂Ω v^2
    """
    kappa = self.kappa
    inv_eps = 1.0 / self.eps

    def norm(v: FunctionState, quad) -> jax.Array:
      grad_sq = jnp.sum(v.grad_interior ** 2, axis=2)           # (NΩ, n_v)
      a1 = jnp.einsum("n,ni->i", quad.interior_w, grad_sq)      # (n_v,)
      b_sq = v.boundary ** 2                                    # (N∂, n_v)
      a2 = jnp.einsum("n,ni->i", quad.boundary_w, b_sq)         # (n_v,)
      en2 = kappa * a1 + inv_eps * a2
      en2 = jnp.maximum(en2, jnp.array(0., en2.dtype))
      return jnp.sqrt(en2)                                      # (n_v,)
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
bounds = ((0.0, 1.0), (0.0, 1.0))
ng = 256
quad = gauss_lobatto_rectangle_quadrature(bounds=bounds, ng=ng)
eps = 1e-4
pde = PoissonUnitSquareRobinBC(eps=eps)
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


# %% Plotting
u_num_fn = make_u_fn(sigma_net_fn_list, u_coeff, basis_coeff_list)

def u_exact_fn(X: jax.Array):
  x = X[:, 0]
  y = X[:, 1]
  c = jnp.asarray(pde.c, dtype=X.dtype)
  return (jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y) + c).reshape(-1, 1)

fig, ax = compare_num_exact_2d(quad.interior_x, u_num_fn, u_exact_fn, kind="tri", error_kind="relative", savepath="cmp2d_tri.png")

# %%
