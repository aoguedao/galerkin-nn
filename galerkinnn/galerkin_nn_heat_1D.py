# %%
import jax
import jax.numpy as jnp
import numpy as np
import optax
import time

from functools import partial
from jax import lax
from jaxtyping import PRNGKeyArray
from typing import Callable, Sequence, Tuple

jax.config.update("jax_enable_x64", True)

eps = 1e-3

# Neural Network
def single_net(
  X: jax.Array,
  params: optax.Params,
  activation: Callable[[jax.Array], jax.Array]
) -> jax.Array:
  X = jnp.dot(X, params["W"]) + params["b"]
  X = activation(X)
  return X


def net_proj(
  X: jax.Array,
  params: optax.Params,
  activation: Callable[[jax.Array], jax.Array],
  coeff: jax.Array
) -> jax.Array:
  """
  Linear combination of the neural network
  """
  net = single_net(X=X, params=params, activation=activation)
  return jnp.dot(net, coeff)


dsingle_net = jax.vmap(jax.jacobian(single_net, argnums=0), in_axes=(0, None, None))
dnet_proj = jax.vmap(jax.jacobian(net_proj, argnums=0), in_axes=(0, None, None, None))


# PDE
def gauss_lengendre_quad(bounds: tuple, n: int) -> Tuple[jax.Array, jax.Array]:
  a, b = bounds
  x, w = np.polynomial.legendre.leggauss(deg=n)
  x = 0.5 * (b - a) * x + 0.5 * (b + a)  # From [-1, 1] to [a, b]
  w = 0.5 * (b - a) * w  # Scale quadrature weights
  return jnp.array(x), jnp.array(w)


def inner_product(u: jax.Array, v: jax.Array, XW: jax.Array) -> float:
  return jnp.sum(XW * u * v)


def norm(
  v: jax.Array,
  dv: jax.Array,
  v_bdry: jax.Array,
  XW: jax.Array,
  XW_bdry: jax.Array
) -> float:
  """
  Norm |||v|||
  """
  a = bilinear_op(
    u=v,
    v=v,
    du=dv,
    dv=dv,
    u_bdry=v_bdry,
    v_bdry=v_bdry,
    XW=XW,
    XW_bdry=XW_bdry
  )
  return jnp.sqrt(a)


def residual(
  u: jax.Array,
  du: jax.Array,
  u_bdry: jax.Array,
  v: jax.Array,
  dv: jax.Array,
  v_bdry: jax.Array,
  f: jax.Array,
  XW: jax.Array,
  XW_bdry: jax.Array
) -> float:
  """
  Residual L(v) - a(u, v)
  """
  L_v = linear_op(f=f, v=v, XW=XW)
  a_uv = bilinear_op(
    u=u,
    v=v,
    du=du,
    dv=dv,
    u_bdry=u_bdry,
    v_bdry=v_bdry,
    XW=XW,
    XW_bdry=XW_bdry
  )
  return L_v - a_uv


def error_eta(
  u: jax.Array,
  du: jax.Array,
  u_bdry: jax.Array,
  v: jax.Array,
  dv: jax.Array,
  v_bdry: jax.Array,
  f: jax.Array,
  XW: jax.Array,
  XW_bdry: jax.Array
) -> float:
  r"""Error approximation
  $$ \eta(u, v) = <r(u), v> / |||v||| = (L(v) - a(u, v)) / |||v||| $$
  """
  norm_v = norm(
    v=v,
    dv=dv,
    v_bdry=v_bdry,
    XW=XW,
    XW_bdry=XW_bdry
  )
  res = residual(
    u=u,
    du=du,
    u_bdry=u_bdry,
    v=v,
    dv=dv,
    v_bdry=v_bdry,
    f=f,
    XW=XW,
    XW_bdry=XW_bdry
  )
  return res / norm_v


def solution_proj(
  coeff: jax.Array,
  bases: Sequence[jax.Array],
) -> jax.Array:
  bases = jnp.stack(bases, axis=1)
  return jnp.dot(bases, coeff)


def solution_pred(
  X: jax.Array,
  coeff: Sequence[jax.Array],
  basis_fns: Sequence[Callable],
) -> jax.Array:
  assert len(basis_fns) == len(coeff)
  bases = [basis_fn(X) for basis_fn in basis_fns]
  u = solution_proj(coeff=coeff, bases=bases)
  return u


# Galerkin Schemes
def galerkin_solve(
  bases: Sequence[jax.Array],
  bases_bdry: Sequence[jax.Array],
  dbases: Sequence[jax.Array],
  f: jax.Array,
  XW: jax.Array,
  XW_bdry: jax.Array,
) -> jax.Array:
  bases = jnp.stack(bases, axis=1)
  dbases = jnp.stack(dbases, axis=1)
  bases_bdry = jnp.stack(bases_bdry, axis=1)

  F = jax.vmap(lambda v : linear_op(f=f, v=v, XW=XW), in_axes=1)(bases)
  K = jax.vmap(
    lambda phi_i, dphi_i, phi_bdry_i: jax.vmap(
      lambda phi_j, dphi_j, phi_bdry_j: bilinear_op(
        u=phi_i,
        du=dphi_i,
        u_bdry=phi_bdry_i,
        v=phi_j,
        dv=dphi_j,
        v_bdry=phi_bdry_j,
        XW=XW,
        XW_bdry=XW_bdry
      ),
      in_axes=1
    )(bases, dbases, bases_bdry),
    in_axes=1
  )(bases, dbases, bases_bdry)
  sol_coeff, _, _, _ = jnp.linalg.lstsq(K, F) # Get solution coefficients
  return sol_coeff


def galerkin_lsq(
  u: jax.Array,
  du: jax.Array,
  u_bdry: jax.Array,
  net: jax.Array,
  dnet: jax.Array,
  net_bdry: jax.Array,
  f: jax.Array,
  XW: jax.Array,
  XW_bdry: jax.Array
) -> jax.Array:

  F = jax.vmap(
    lambda v, dv, v_bdry: residual(
      u=u, du=du, u_bdry=u_bdry, v=v, dv=dv, v_bdry=v_bdry, f=f, XW=XW, XW_bdry=XW_bdry
    ),
    in_axes=1
  )(net, dnet, net_bdry)

  K = jax.vmap(
    lambda v_i, dv_i, v_bdry_i: jax.vmap(
      lambda v_j, dv_j, v_bdry_j: bilinear_op(
        u=v_i,
        du=dv_i,
        u_bdry=v_bdry_i,
        v=v_j,
        dv=dv_j,
        v_bdry=v_bdry_j,
        XW=XW,
        XW_bdry=XW_bdry
      ),
      in_axes=1
    )(net, dnet, net_bdry),
    in_axes=1
  )(net, dnet, net_bdry)
  # print(K.shape)
  # print(F.shape)
  coeff, _, _, _ = jnp.linalg.lstsq(K, F)
  return coeff

def loss_fn(
  params: optax.Params,
  u: jax.Array,
  du: jax.Array,
  u_bdry: jax.Array,
  f: jax.Array,
  X: jax.Array,
  X_bdry: jax.Array,
  XW: jax.Array,
  XW_bdry: jax.Array,
  activation: Callable[[jax.Array], jax.Array],
) -> tuple[float, jax.Array]:
  # Net with input layer and hidden layer
  net = single_net(X=X.reshape(-1, X.ndim), params=params, activation=activation)
  net_bdry = single_net(X=X_bdry.reshape(-1, X.ndim), params=params, activation=activation)
  dnet = dsingle_net(X.reshape(-1, X.ndim), params, activation).squeeze()
  # Get output layer coefficients
  v_nn_coeff = galerkin_lsq(
    u=u,
    du=du,
    u_bdry=u_bdry,
    net=net,
    dnet=dnet,
    net_bdry=net_bdry,
    f=f,
    XW=XW,
    XW_bdry=XW_bdry,
  )
  v_nn = net_proj(X=X.reshape(-1, X.ndim), params=params, activation=activation, coeff=v_nn_coeff)
  v_nn_bdry = net_proj(X=X_bdry.reshape(-1, X.ndim), params=params, activation=activation, coeff=v_nn_coeff)
  dv_nn = dnet_proj(X.reshape(-1, X.ndim), params, activation, v_nn_coeff).squeeze()
  # Loss
  loss = error_eta(
    u=u,
    du=du,
    u_bdry=u_bdry,
    v=v_nn,
    dv=dv_nn,
    v_bdry=v_nn_bdry,
    f=f,
    XW=XW,
    XW_bdry=XW_bdry
  )
  return -jnp.abs(loss), v_nn_coeff  # It is maximizing!

@partial(jax.jit, static_argnames=["optimizer", "activation"])
def train_step(
  optimizer: optax.GradientTransformation,
  opt_state: optax.OptState,
  params: optax.Params,
  u: jax.Array,
  du: jax.Array,
  u_bdry: jax.Array,
  f: jax.Array,
  X: jax.Array,
  X_bdry: jax.Array,
  XW: jax.Array,
  XW_bdry: jax.Array,
  activation: Callable[[jax.Array], jax.Array],
) -> tuple[optax.OptState, optax.Params, float, jax.Array]:
  (loss, coeff), grads = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)(
    params,
    u,
    du,
    u_bdry,
    f,
    X,
    X_bdry,
    XW,
    XW_bdry,
    activation
  )
  updates, opt_state = optimizer.update(grads, opt_state, params)
  params = optax.apply_updates(params, updates)
  return opt_state, params, loss, coeff, grads


def make_phi_fn(params, activation, v_nn_coeff):
  """Create a phi function with fixed parameters."""
  def phi_fn(x):
    return net_proj(
      X=x.reshape(-1, x.ndim),
      params=params,
      activation=activation,
      coeff=v_nn_coeff
    )
  return phi_fn

def make_dphi_fn(params, activation, v_nn_coeff):
  """Create a phi function with fixed parameters."""
  def dphi_fn(x):
    return dnet_proj(x.reshape(-1, x.ndim), params, activation, v_nn_coeff).squeeze()
  return dphi_fn

def augment_basis(
  u: jax.Array,
  du: jax.Array,
  u_bdry: jax.Array,
  f: jax.Array,
  X: jax.Array,
  X_bdry: jax.Array,
  XW: jax.Array,
  XW_bdry: jax.Array,
  activation: Callable[[jax.Array], jax.Array],
  neurons: int,
  learning_rate: float,
  max_epoch: int,
  tol_basis: float,
  key: PRNGKeyArray,
):
  print(f"d/dx Activation(0): {jax.grad(activation)(0.0)}")
  print(f"Neurons: {neurons}")
  print(f"Learning Rate: {learning_rate}")
  optimizer = optax.adam(learning_rate=learning_rate)
  key_W, key_b = jax.random.split(key, num=2)
  # initializer_W = jax.nn.initializers.glorot_normal()
  params = {
    # "W": jnp.ones(shape=(1, neurons)),
    # "W": initializer_W(key=key_W, shape=(1, neurons)),
    "W": jax.random.normal(shape=(1, neurons), key=key_W),
    "b": - jnp.linspace(0, 1, neurons),
  }
  # (loss, coeff), grads = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)(
  # 	params,
  # 	u,
  # 	du,
  # 	u_bdry,
  # 	f,
  # 	X,
  # 	X_bdry,
  # 	XW,
  # 	XW_bdry,
  # 	activation
  # )
  # print(f"Initial loss: {loss:.4e}, grads: {grads}")
  opt_state = optimizer.init(params)
  loss_prev = 1
  for i in range(max_epoch):
    opt_state, params, loss, coeff, grads = train_step(
      optimizer=optimizer,
      opt_state=opt_state,
      params=params,
      u=u,
      du=du,
      u_bdry=u_bdry,
      f=f,
      X=X,
      X_bdry=X_bdry,
      XW=XW,
      XW_bdry=XW_bdry,
      activation=activation,
    )
    if i % (max_epoch // 10) == 0:
      print(f"step {i}, loss: {- float(loss)}, grad norm: {optax.global_norm(grads):.4e}")
    if jnp.abs((loss - loss_prev) / loss_prev) < tol_basis:
      break
    else:
      loss_prev = loss

  # Get the final basis
  net = single_net(X=X.reshape(-1, X.ndim), params=params, activation=activation)
  net_bdry = single_net(X=X_bdry.reshape(-1, X.ndim), params=params, activation=activation)
  dnet = dsingle_net(X.reshape(-1, X.ndim), params, activation).squeeze()
  v_nn_coeff = galerkin_lsq(
    u=u,
    du=du,
    u_bdry=u_bdry,
    net=net,
    dnet=dnet,
    net_bdry=net_bdry,
    f=f,
    XW=XW,
    XW_bdry=XW_bdry,
  )
  # Basis $\phi^{NN}$
  # X_int = X
  # def basis_fn(
  # 	X: jax.Array = X,
  # 	X_bdry: jax.Array = X_bdry,
  # 	XW: jax.Array = XW,
  # 	XW_bdry: jax.Array = XW_bdry,
  # 	params: optax.Params = params,
  # 	activation: Callable[[jax.Array], jax.Array] = activation,
  # 	coeff: jax.Array = v_nn_coeff,
  # ):
  # 	# v_nn = net_proj(X=X.reshape(-1, X.ndim), params=params, coeff=coeff, activation=activation)
  # 	# v_nn_bdry = net_proj(X=X_bdry.reshape(-1, X.ndim), params=params, activation=activation, coeff=coeff)
  # 	# dv_nn = dnet_proj(X.reshape(-1, X.ndim), params, activation, coeff).squeeze()
  # 	# v_nn_norm = norm(v=v_nn, dv=dv_nn, v_bdry=v_nn_bdry, XW=XW, XW_bdry=XW_bdry)
  # 	v_nn_norm = 1

  # 	@jax.jit
  # 	def phi(x):
  # 		return net_proj(X=x.reshape(-1, x.ndim), params=params, coeff=coeff, activation=activation) / v_nn_norm

  # 	@jax.jit
  # 	def dphi(x):
  # 		return dnet_proj(x.reshape(-1, x.ndim), params, activation, coeff).squeeze() / v_nn_norm

  # 	return phi, dphi

  # @jax.jit
  # def phi_fn(x):
  # 	return net_proj(X=x.reshape(-1, x.ndim), params=params, activation=activation, coeff=v_nn_coeff)

  # def dphi_fn(x):
  # 	return dnet_proj(x.reshape(-1, x.ndim), params, activation, v_nn_coeff).squeeze()
  phi_fn = make_phi_fn(params, activation, v_nn_coeff)
  dphi_fn = make_dphi_fn(params, activation, v_nn_coeff)

  # phi_fn, dphi_fn = basis_fn()
  phi_nn = phi_fn(X)
  phi_nn_bdry = phi_fn(X_bdry)
  dphi_nn = dphi_fn(X)
  eta = error_eta(
    u=u,
    du=du,
    u_bdry=u_bdry,
    v=phi_nn,
    dv=dphi_nn,
    v_bdry=phi_nn_bdry,
    f=f,
    XW=XW,
    XW_bdry=XW_bdry
  )

  return phi_nn, phi_nn_bdry, dphi_nn, eta, params, coeff, phi_fn, dphi_fn


def adaptive_subspace(
  xbounds: Tuple[float, float],
  source: Callable[[jax.Array], jax.Array],
  # linear_op: Callable[[jax.Array], jax.Array],
  # bilinear_op: Callable[[jax.Array], jax.Array],
  u0: Callable[[jax.Array], jax.Array],
  du0: Callable[[jax.Array], jax.Array],
  n_train: int,
  n_val: int,
  activations_fn: Callable[[int], Callable[[jax.Array], jax.Array]],
  network_widths_fn: Callable[[int], int],
  learning_rates_fn: Callable[[int], float],
  max_bases: int = 8,
  max_epoch_basis: int = 5_000,
  tol_solution: float = 1e-8,
  tol_basis: float = 1e-6,
  seed: int = 42
):
  # Generate data
  key = jax.random.key(seed)
  xa, xb = xbounds
  X_train, XW_train = gauss_lengendre_quad((xa, xb), n_train)
  X_val, XW_val = gauss_lengendre_quad((xa, xb), n_val)
  X_bdry = jnp.array([xa, xb])
  XW_bdry = jnp.array([1.0, 1.0])  # Hardcoded for now
  f_train = source(X_train)

  eta_errors = []  # Remember, for $\eta_i$ we need $\phi_{i+1}^{NN}$
  bases_params = []
  bases_coeffs = []
  bases_train = []  # $\phi_i^{NN}(X_train)$
  bases_bdry = []
  dbases_train = []
  basis_fns = []
  dbasis_fns = []
  solution_coeffs = []

  u_train = u0(X_train)
  du_train = du0(X_train)
  u_bdry = u0(X_bdry)


  # Basis step loop
  bstep = 1
  error = 1e10
  while (error > tol_solution) and (bstep <= max_bases):
    print(f"Basis Step: {bstep}")

    activation = activations_fn(bstep)
    neurons = network_widths_fn(bstep)
    learning_rate = learning_rates_fn(bstep)
    key, _ = jax.random.split(key, num=2)

    phi_nn, phi_nn_bdry, dphi_nn, eta, params, coeff, phi_nn_fn, dphi_nn_fn = augment_basis(
      u=u_train,
      du=du_train,
      u_bdry=u_bdry,
      f=f_train,
      X=X_train,
      X_bdry=X_bdry,
      XW=XW_train,
      XW_bdry=XW_bdry,
      activation=activation,
      neurons=neurons,
      learning_rate=learning_rate,
      max_epoch=max_epoch_basis,
      tol_basis=tol_basis,
      key=key,
    )

    eta_errors.append(eta)
    bases_params.append(params)
    bases_coeffs.append(coeff)
    bases_train.append(phi_nn)
    bases_bdry.append(phi_nn_bdry)
    dbases_train.append(dphi_nn)
    basis_fns.append(phi_nn_fn)
    dbasis_fns.append(dphi_nn_fn)

    # Solution coefficients
    u_coeff = galerkin_solve(
      bases_train,
      bases_bdry,
      dbases_train,
      f_train,
      XW_train,
      XW_bdry,
    )
    print(f"Solution coeffs: {u_coeff}")
    solution_coeffs.append(u_coeff)
    u_train = solution_proj(u_coeff, bases=bases_train)
    u_bdry = solution_proj(u_coeff, bases=bases_bdry)
    du_train = solution_proj(u_coeff, bases=dbases_train)

    bstep += 1
    error = jnp.abs(eta)
    print(f"Eta error: {error}")

  return (
    eta_errors,
    solution_coeffs,
    bases_params,
    bases_coeffs,
    bases_train,
    bases_bdry,
    dbases_train,
    basis_fns,
    dbasis_fns
  )

def make_source(t_step, u_prev):
  def source(X: jax.Array):
    return t_step * 2 * jnp.sin(jnp.pi * X) - u_prev(X)
  return source

def linear_op(f: jax.Array, v: jax.Array, XW: jax.Array) -> float:
  return inner_product(u=f, v=v, XW=XW)

def bilinear_op(
  u: jax.Array,
  du: jax.Array,
  u_bdry: jax.Array,
  v: jax.Array,
  dv: jax.Array,
  v_bdry: jax.Array,
  XW: jax.Array,
  XW_bdry: jax.Array
) -> float:
  a1 = inner_product(u=du, v=dv, XW=XW)
  a2 = inner_product(u=u, v=v, XW=XW)
  a3 = inner_product(u=u_bdry, v=v_bdry, XW=XW_bdry)
  # eps = 1e-3
  return t_step * a1 + a2 + a3 / eps
# %%
if __name__ == "__main__":
  # PDE
  xbounds = 0.0, 1.0
  tbounds = 0.0, 1.0
  u_initial = lambda x: 0.0
  t_step = 0.01

  # NN
  n_train = 512
  n_val = 1024
  N = 5  # Init Neurons
  r = 2  # Neurons Growth
  A = 5 * 1e-3  # Init Learning Rate
  rho = 1.1  # Learning Rate Growth

  def activations_fn(i):
    scale_fn = lambda i: i
    scale_i = scale_fn(i)
    def activation(x):
      return jnp.tanh(scale_i * x)
    return activation

  network_widths_fn = lambda i: N * r ** (i - 1)
  learning_rates_fn = lambda i: A * rho ** (-(i - 1))

  max_bases = 6
  max_epoch_basis = 50
  tol_solution = 1e-9
  tol_basis = 1e-9
  seed = 42


  # def solution(X: jax.Array):
  #   u1 = jnp.sin(2 * jnp.pi * X)
  #   u2 = 0.1 * jnp.sin(25 * jnp.pi * X)
  #   u3 = 5.0 * jnp.pi * eps * (10 * eps - 8 * X + 9) / (20 * eps + 10)
  #   return u1 + u2 + u3

  # n_test = 128
  # X_test, XW_test = gauss_lengendre_quad(xbounds, n_test)
  # X_bdry = jnp.array(xbounds, dtype=float)
  # XW_bdry = jnp.array([1.0, 1.0])

  # u_actual = solution(X_test)
  # u_pred = solution_pred(X=X_test, coeff=solution_coeffs[-1], basis_fns=basis_fns)
  # print(jnp.linalg.norm(u_actual - u_pred))


  # import matplotlib.pyplot as plt
  # fig, ax = plt.subplots()
  # ax.plot(X_test, u_actual, label="actual")
  # ax.plot(X_test, u_pred, label="estimated")
  # ax.legend()
  # fig.show()
  # fig.savefig("solution_heat.png")



# %%

t_array = np.arange(tbounds[0], tbounds[1], t_step) + t_step
u_prev = u_initial
for t in t_array:
  source = make_source(t_step, u_prev)
  u0 = lambda X: jnp.zeros_like(X)
  du0 = lambda X: jnp.zeros_like(X)
  start = time.perf_counter()
  (
    eta_errors,
    solution_coeffs,
    bases_params,
    bases_coeffs,
    bases_train,
    bases_bdry,
    dbases_train,
    basis_fns,
    dbasis_fns
  ) = adaptive_subspace(
    xbounds=xbounds,
    source=source,
    u0=u0,
    du0=du0,
    n_train=n_train,
    n_val=n_val,
    activations_fn=activations_fn,
    network_widths_fn=network_widths_fn,
    learning_rates_fn=learning_rates_fn,
    max_bases=max_bases,
    max_epoch_basis=max_epoch_basis,
    tol_solution=tol_solution,
    tol_basis=tol_basis,
    seed=seed,
  )
  end = time.perf_counter()
  elapsed = end - start
  print(f"Elapsed time: {elapsed:.6f} seconds")
  def make_solution(coeff, basis_fns):
    def solution(X: jax.Array):
      bases = [basis_fn(X) for basis_fn in basis_fns]
      u = solution_proj(coeff=coeff, bases=bases)
      return u
    return solution

  u_prev = make_solution(coeff=solution_coeffs[-1], basis_fns=basis_fns)
  break