import jax
import jax.numpy as jnp
import optax

from typing import Callable
from functools import partial

import galerkin_nn as glknn
from galerkin_nn.pde import PDE

jax.config.update("jax_enable_x64", True)


@jax.tree_util.register_dataclass
class Poisson1D(PDE):
  def source(self) -> Callable:
    def _source(X: jax.Array) -> jax.Array:
      f1 = (2 * jnp.pi) ** 2 * jnp.sin(2 * jnp.pi * X)
      f2 = 0.1 * (25 * jnp.pi) ** 2 * jnp.sin(25 * jnp.pi * X)
      return f1 + f2
    return _source

  def inner_product(self) -> Callable:
    def _inner_product(u: jax.Array, v: jax.Array, XW: jax.Array) -> float:
      return jnp.sum(XW * u * v)
    return _inner_product

  def linear_operator(self) -> Callable:
    _inner_product = self.inner_product()
    def _linear_operator(f: jax.Array, v: jax.Array, XW: jax.Array) -> float:
      return _inner_product(u=f, v=v, XW=XW)
    return _linear_operator

  def bilinear_form(self) -> Callable:
    eps = 1e-3
    _inner_product = self.inner_product()
    def _bilinear_form(
      u: jax.Array,
      du: jax.Array,
      u_bdry: jax.Array,
      v: jax.Array,
      dv: jax.Array,
      v_bdry: jax.Array,
      XW: jax.Array,
      XW_bdry: jax.Array
    ) -> float:
      a1 = _inner_product(u=du, v=dv, XW=XW)
      a2 = _inner_product(u=u_bdry, v=v_bdry, XW=XW_bdry)
      return a1 + a2 / eps
    return _bilinear_form


def single_net(
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


def make_net_fn(activation):
  def net(X, params):
    return single_net(X, params, activation)
  return net

# Parameters
seed = 42
N = 5  # Init Neurons
r = 2  # Neurons Growth
A = 5 * 1e-3  # Init Learning Rate
rho = 1.1  # Learning Rate Growth
network_widths_fn = lambda i: N * r ** (i - 1)
learning_rates_fn = lambda i: A * rho ** (-(i - 1))

xa, xb = 0.0, 1.0
u0 = lambda X: jnp.zeros_like(X)
du0 = lambda X: jnp.zeros_like(X)
max_bases = 4
quad_name = "gauss-legendre"
num_train = 512
num_val = 1024
max_epoch_per_basis = 50
tol_solution = 1e-9
tol_basis = 1e-9

key = jax.random.key(seed)
net_fns = []
init_net_params = []
optimizers = []
for i in range(1, max_bases + 1):
  activation = activations_fn(i)
  neurons = network_widths_fn(i)
  lr = learning_rates_fn(i)
  key_W, key = jax.random.split(key, num=2)
  net_fn = make_net_fn(activation=activation)
  params = {
		"W": jnp.ones(shape=(1, neurons)),
		# "W": jax.random.normal(shape=(1, neurons), key=key_W),
		"b": - jnp.linspace(0, 1, neurons),
	}
  optimizer = optax.adam(learning_rate=lr)
  net_fns.append(net_fn)
  init_net_params.append(params)
  optimizers.append(optimizer)
  # print(f"Neurons: {neurons}")
  # print(f"Learning Rate: {lr}")
  # print(f"d/dx Activation(0): {jax.grad(activation)(0.0)}")

domain = glknn.domain.IntervalGeom(x_start=xa, x_end=xb)
pde = Poisson1D(domain=domain)
solver = glknn.solvers.GalerkinNN1D(pde=pde)
solver.fit(
  u0=u0,
  du0=du0,
  max_bases=max_bases,
  net_fns=net_fns,
  init_net_params=init_net_params,
  optimizers=optimizers,
  quad_name=quad_name,
  num_train=num_train,
  num_val=num_val,
  max_epoch_per_basis=max_epoch_per_basis,
  tol_solution=tol_solution,
  tol_basis=tol_basis,
)

def solution(X: jax.Array):
  eps = 1e-3
  u1 = jnp.sin(2 * jnp.pi * X)
  u2 = 0.1 * jnp.sin(25 * jnp.pi * X)
  u3 = 5.0 * jnp.pi * eps * (10 * eps - 8 * X + 9) / (20 * eps + 10)
  return u1 + u2 + u3

num_test = 1024
X_test, XW_test = solver.pde.domain._quadrature_interior(degree=num_test, name="gauss-legendre")
u_actual = solution(X_test)
u_pred = solver.solution_fn()(X_test[:, jnp.newaxis])
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(X_test, u_actual, label="actual")
ax.plot(X_test, u_pred, label="estimated")
ax.legend()
fig.show()
fig.savefig("solution.png")
# # %%
# (Array(-4.46557945, dtype=float64), Array([ -905.81533359,  3048.63007825, -4313.22896374,  3048.96388438,
#         -906.0070005 ], dtype=float64))

# net coeff first iteration
# Array([ -296.91652829,  2076.7686476 , -3855.34857611,  2487.96204111,
#         -466.54647927], dtype=float64)