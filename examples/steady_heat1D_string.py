# steady_heat1D_robin.py
# %%
import jax
import jax.numpy as jnp
import optax
from typing import Callable

from flax import struct

from galerkinnn import FunctionState, PDE, Quadrature, GalerkinNN
from galerkinnn.quadratures import interval_quadrature

# -------------------------
# Hyper-parameters
# -------------------------
seed = 42
max_bases = 5
max_epoch_basis = 100
tol_solution = 1e-7
tol_basis = 1e-7

N = 5    # init neurons
r = 2    # neurons growth
A = 5e-3 # init learning rate
rho = 1.1

# -------------------------
# PDE: Steady 1D heat with general Robin BC
# -(k u')' = f  in (0, L)
# k ∂_n u + h(x_b) u = g(x_b)  on {0, L}
#
# Weak form:
#   a(u,v) = ∫ k u' v' dx + Σ_b h(x_b) u(x_b) v(x_b)
#   L(v)   = ∫ f v dx     + Σ_b g(x_b) v(x_b)
# -------------------------
@struct.dataclass
class SteadyHeat1DRobin(PDE):
  # Mark callables as static leaves (non-pytree) so JAX won't try to trace them.
  k_fn:      Callable = struct.field(pytree_node=False)  # (N,dim)->(N,1), k(x) >= k_min > 0
  f_fn:      Callable = struct.field(pytree_node=False)  # (N,dim)->(N,1)
  h_bdry_fn: Callable = struct.field(pytree_node=False)  # (Nb,dim)->(Nb,1)
  g_bdry_fn: Callable = struct.field(pytree_node=False)  # (Nb,dim)->(Nb,1)

  def k(self, X: jax.Array) -> jax.Array:
    return self.k_fn(X)

  def source(self) -> Callable[[jax.Array], jax.Array]:
    return self.f_fn

  # L(v) = (f,v)_Ω + (g,v)_∂Ω
  def linear_operator(self) -> Callable:
    f = self.source()
    g_b = self.g_bdry_fn
    def L(v: FunctionState, quad: Quadrature) -> jnp.ndarray:
      fvals = f(quad.interior_x).reshape(-1)                 # (N,)
      Lin = jnp.sum((fvals[:, None] * v.interior) * quad.interior_w[:, None],
                    axis=0, keepdims=True)                  # (1, n_v)
      gvals = g_b(quad.boundary_x).reshape(-1)               # (Nb,)
      Lbd = jnp.sum((gvals[:, None] * v.boundary) * quad.boundary_w[:, None],
                    axis=0, keepdims=True)                  # (1, n_v)
      return Lin + Lbd
    return L

  # a(u,v) = (k u', v')_Ω + (h u, v)_∂Ω
  def bilinear_form(self) -> Callable:
    k = self.k
    h_b = self.h_bdry_fn
    def a(u: FunctionState, v: FunctionState, quad: Quadrature) -> jnp.ndarray:
      kvals = k(quad.interior_x).reshape(-1)                 # (N,)
      a1 = jnp.einsum("nui,nvi,n->uv",
                      u.grad_interior, v.grad_interior, kvals * quad.interior_w)
      hvals = h_b(quad.boundary_x).reshape(-1)               # (Nb,)
      a2 = jnp.einsum("an,am,a->nm",
                      u.boundary, v.boundary, hvals * quad.boundary_w)
      return a1 + a2
    return a

  # ||v||_a^2 = ∫ k |v'|^2 + Σ h v^2(boundary)
  def energy_norm(self) -> Callable:
    k = self.k
    h_b = self.h_bdry_fn
    def norm(v: FunctionState, quad: Quadrature) -> jax.Array:
      kvals = k(quad.interior_x).reshape(-1)                 # (N,)
      grad_sq = jnp.sum(v.grad_interior**2, axis=2)          # (N, n_v)
      a1 = jnp.einsum("n,ni->i", kvals * quad.interior_w, grad_sq)
      hvals = h_b(quad.boundary_x).reshape(-1)               # (Nb,)
      b_sq = v.boundary**2                                   # (Nb, n_v)
      a2 = jnp.einsum("n,ni->i", hvals * quad.boundary_w, b_sq)
      en2 = jnp.maximum(a1 + a2, jnp.array(0., v.interior.dtype))
      return jnp.sqrt(en2)                                    # (n_v,)
    return norm

# -------------------------
# Neural network basis (same style as your 1D example)
# -------------------------
def net_fn(
  X: jax.Array,
  params: optax.Params,
  activation: Callable[[jax.Array], jax.Array]
) -> jax.Array:
  X = jnp.dot(X, params["W"]) + params["b"]
  return activation(X)

def activations_fn(i: int):
  return lambda x: jnp.tanh(i * x)

network_widths_fn = lambda i: N * r ** (i - 1)
learning_rates_fn = lambda i: A * rho ** (-(i - 1))

# -------------------------
# Analytical solution for a specific, physically-interesting case
# k(x) = 1 + x,  f(x) = 10,  convective BCs:
#   at x=0:  k * (-u'(0)) + h0 (u(0) - Tinf) = 0
#   at x=1:  k * (+u'(1)) + hL (u(1) - Tinf) = 0
#
# Derivation summary:
#   (k u')' = -10  => u(x) = -10 x + (10 + C1) ln(1+x) + C2
#   C1 = 10 h0 (1 + hL - hL ln 2) / (h0 + hL + h0 hL ln 2)
#   C2 = Tinf + C1 / h0
# -------------------------
def u_exact_k1px_f10(X: jax.Array, h0: float, hL: float, Tinf: float) -> jax.Array:
  x = X[:, 0:1]
  ln2 = jnp.log(2.0)
  denom = h0 + hL + h0 * hL * ln2
  C1 = 10.0 * h0 * (1.0 + hL - hL * ln2) / denom
  C2 = Tinf + C1 / h0
  return -10.0 * x + (10.0 + C1) * jnp.log(1.0 + x) + C2

# -------------------------
# Build quadrature, PDE, and solve
# -------------------------
xbounds = (0.0, 1.0)
n_train = 256
quad = interval_quadrature(xbounds, n_train)

# Choose the analytic-comparable case:
k  = lambda X: 1.0 + X[:, 0:1]               # (N,1)
f  = lambda X: 10.0 * jnp.ones_like(X[:, 0:1])

# Robin data on boundary: h(0)=h0, h(1)=hL; g = h * Tinf (convection to ambient)
h0, hL = 5.0, 5.0
Tinf = 0.25
h_bdry = lambda Xb: jnp.array([h0, hL]).reshape(-1, 1)              # (2,1)
g_bdry = lambda Xb: jnp.array([h0*Tinf, hL*Tinf]).reshape(-1, 1)    # (2,1)

pde = SteadyHeat1DRobin(k_fn=k, f_fn=f, h_bdry_fn=h_bdry, g_bdry_fn=g_bdry)

# Zero initial state
u0_fn = lambda X: jnp.zeros_like(X[:, 0:1])   # (N,1)
u0_grad = lambda X: jnp.zeros_like(X)         # (N,dim)
u0 = FunctionState.from_function(func=u0_fn, quad=quad, grad_func=u0_grad)

# -------------------------
# Solve with GalerkinNN
# -------------------------
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

elapsed = time.perf_counter() - start
print(f"Elapsed time: {elapsed:.6f} s")

# -------------------------
# Compare against analytical solution
# -------------------------
import matplotlib.pyplot as plt

u_actual = u_exact_k1px_f10(quad.interior_x, h0=h0, hL=hL, Tinf=Tinf)  # (N,1)
u_pred = u.interior                                                    # (N,1)

fig, ax = plt.subplots()
ax.plot(quad.interior_x[:, 0], u_actual[:, 0], label="Exact", lw=2)
ax.plot(quad.interior_x[:, 0], u_pred[:, 0],   label="NN", ls="--")
ax.set_xlabel("x")
ax.set_ylabel("Temperature")
ax.legend()
ax.grid(True)
fig.savefig("steady_heat1d_robin.png", dpi=160)

# L2 error on the quadrature points (simple diagnostic)
l2_err = jnp.sqrt(jnp.mean((u_pred - u_actual)**2))
print("L2 error:", float(l2_err))
