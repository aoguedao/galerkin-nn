# %%  
import jax
import jax.numpy as jnp
import optax

from flax import struct
from typing import Callable

from galerkinnn import FunctionState, PDE, Quadrature, GalerkinNN
from galerkinnn.quadratures import interval_quadrature
from galerkinnn.utils import make_u_fn

xbounds = (0.0, 1.0)
eps = 1e-3

# Hyper-parameters
seed = 42
max_bases = 8
max_epoch_basis = 50
tol_solution = 2e-6
tol_basis = 1e-7  

n_train = 512 
n_val = 1024
N = 5  # Init Neurons
r = 2  # Neurons Growth
A = 5e-2  # Init Learning Rate
rho = 1.1  # Learning Rate Growth

# PDE
@struct.dataclass
class PoissonStringDisplacementRobinBC(PDE):
  """1D Poisson (string displacement) with Robin BC:
    -u'' = f  in (0,1)
    u + eps * n·∂u = 0 on boundary
    bilinear form: a(u,v) = (u', v')_Ω + eps^{-1} (u(0)v(0) + u(1)v(1))
  """
  eps: float = 1e-4

  def source(self):
    def f(X: jax.Array) -> jax.Array:
      f1 = (2 * jnp.pi) ** 2 * jnp.sin(2 * jnp.pi * X)
      f2 = (4 * jnp.pi) ** 2 * jnp.sin(4 * jnp.pi * X)
      f3 = (6 * jnp.pi) ** 2 * jnp.sin(6 * jnp.pi * X)
      return f1 + f2 + f3
    return f

  def linear_operator(self):
    f = self.source()
    def L(v: FunctionState, quad: Quadrature) -> jnp.ndarray:
      fvals = f(quad.interior_x)
      integrand = fvals * v.interior       # (N, n_v)
      return quad.integrate_interior(integrand)  # (1, n_v)
    return L

  def bilinear_form(self):
    eps = self.eps
    def a(u: FunctionState, v: FunctionState, quad: Quadrature) -> jnp.ndarray:
      # u.grad_interior: (N, n_u, dim)
      # v.grad_interior: (N, n_v, dim)
      # interior term: sum_n sum_d  (w_n * u'_{n,:,d} @ v'_{n,:,d})
      a1 = jnp.einsum("nui,nvi,n->uv", u.grad_interior, v.grad_interior, quad.interior_w)
      # boundary term: sum_a (w_a * u_a v_a)
      a2 = jnp.einsum("an,am,a->nm", u.boundary, v.boundary, quad.boundary_w)  # (n_u, n_v)
      return a1 + (1.0 / eps) * a2
    return a

  def energy_norm(self) -> Callable:
    eps = self.eps
    def norm(v, quad) -> jax.Array:
      # returns (n_v,)
      grad_sq = jnp.sum(v.grad_interior ** 2, axis=2)  # (N, n_v)
      a1 = jnp.einsum('n,ni->i', quad.interior_w, grad_sq)  # (n_v,)
      b_sq = v.boundary ** 2  # (Nb, n_v)
      a2 = jnp.einsum('n,ni->i', quad.boundary_w, b_sq)               # (n_v,)
      en2 = a1 + (1.0 / eps) * a2
      en2 = jnp.maximum(en2, jnp.array(0., en2.dtype))
      return jnp.sqrt(en2)  # (n_v,)
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
  scale_i = i
  def activation(x):
    return jnp.tanh(scale_i * x)
  return activation

network_widths_fn = lambda i: N * r ** (i - 1)
learning_rates_fn = lambda i: A * rho ** (-(i - 1))

# Galerkin Neural Network Solver
quad = interval_quadrature(xbounds, n_train)
pde = PoissonStringDisplacementRobinBC(eps=eps)
u0_fn = lambda X: jnp.zeros_like(X)
u0_grad = lambda X: jnp.zeros_like(X)
u0 = FunctionState.from_function(func=u0_fn, quad=quad, grad_func=u0_grad)
# %%
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
def u_sol(X: jax.Array):
  u1 = jnp.sin(2 * jnp.pi * X)
  u2 = jnp.sin(4 * jnp.pi * X)
  u3 = jnp.sin(6 * jnp.pi * X)
  ur = (-24 * jnp.pi * eps * X  + 12 * jnp.pi * eps) / (1 + 2 * eps)
  return u1 + u2 + u3 + ur

u_actual = u_sol(quad.interior_x)
u_pred = u.interior
# fig, ax = plt.subplots()
# ax.plot(quad.interior_x, u_actual, lw=1.5, label="actual")
# ax.plot(quad.interior_x, u_pred, "--", label="estimated")
# ax.legend()
# fig.savefig("images/poisson1d.png")
err = (u_pred.squeeze() - u_actual.squeeze())              # (N,)
L2_err = jnp.sqrt(jnp.einsum('n,n->', quad.interior_w, err**2))
L2_ref = jnp.sqrt(jnp.einsum('n,n->', quad.interior_w, (u_actual.squeeze())**2))
rel_L2 = L2_err / L2_ref

Linf = jnp.max(jnp.abs(err))

print(f"L2 error      = {L2_err:.6e}")
print(f"relative L2   = {rel_L2:.6e}")
print(f"L-infinity    = {Linf:.6e}")


fig, ax = plt.subplots(figsize=(7.2,4))
ax.plot(quad.interior_x, u_actual,  'k--', lw=2.5, label='Analytic')
ax.plot(quad.interior_x, u_pred, 'C0-',  lw=2, label='GalerkinNN')
ax.set_xlabel("$x$")
ax.set_ylabel("$u(x)$")
ax.set_title(rf"Poisson 1D: String displacement ($\varepsilon$={eps:g})")
ax.legend()
ax.grid(alpha=0.25)
plt.tight_layout()
fig.savefig(f"images/poisson1d_string_eps_{eps}__maxbases_{max_bases}.png")
plt.show()
# %%
u_fn = make_u_fn(sigma_net_fn_list, u_coeff, basis_coeff_list)
error_fn = lambda x: u_fn(x) - u_actual(x)
plt.semilogy(eta_errors)
plt.show()