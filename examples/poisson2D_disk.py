# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

from typing import Callable

from galerkinnn.utils import make_u_fn, compare_num_exact_2d
from galerkinnn import FunctionState, PDE, Quadrature, GalerkinNN
from galerkinnn.quadratures import disk_quadrature

# Hyper-parameters
seed = 42
max_bases = 8
max_epoch_basis = 100
tol_solution = 2e-6
tol_basis = 1e-7

# n_train = 128
n_val = 1024
N = 200  # Init Neurons
r = 100  # Neurons Growth
A = 2e-2  # Init Learning Rate
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
  # scale_i = i
  def activation(x):
    # return jnp.tanh(scale_i * x)
    return jnp.tanh(x)
  return activation

# network_widths_fn = lambda i: N * r ** (i - 1)
network_widths_fn = lambda i: N + r * int((i - 1) / 2)
learning_rates_fn = lambda i: A * rho ** (-(i - 1))


# Galerkin Neural Network Solver
R = 1.0
nr, nt = 128, 128  # radius, angle
quad = disk_quadrature(radius=R, n_r=nr, n_theta=nt)
eps = 1e-4
pde = PoissonMembraneDisplacementRobinBC(eps=eps)
u0_fn = lambda X: jnp.zeros(shape=(X.shape[0], 1))
u0_grad = lambda X: jnp.zeros_like(X)
u0 = FunctionState.from_function(func=u0_fn, quad=quad, grad_func=u0_grad)

#%%
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
# Plotting
u_num_fn = make_u_fn(sigma_net_fn_list, u_coeff, basis_coeff_list)

def u_exact_fn(X: jax.Array):
  r2 = jnp.sum(X**2, axis=1, keepdims=True)
  return -0.5 * r2 + eps + 0.5

# fig, ax = compare_num_exact_2d(quad.interior_x, u_num_fn, u_exact_fn, kind="tri", savepath="cmp2d_tri.png")
plt.close()
plt.semilogy(eta_errors)
