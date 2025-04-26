# %%
import jax
import jax.numpy as jnp
import numpy as np
import optax

from functools import partial
from jaxtyping import PRNGKeyArray
from typing import Callable, Sequence, Tuple

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
	return jnp.array(x).reshape(-1, 1), jnp.array(w).reshape(-1, 1)


def inner_product(u: jax.Array, v: jax.Array, XW: jax.Array) -> float:
	return jnp.sum(XW * u * v)


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
	a2 = inner_product(u=u_bdry, v=v_bdry, XW=XW_bdry)
	eps = 1e-4  # TODO
	return a1 + a2 / eps


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
):
	bases_matrix = jnp.concat(bases, axis=1)
	return jnp.dot(bases_matrix, coeff)


# Galerkin Schemes
# @jax.jit
def galerkin_solve(
	bases: Sequence[jax.Array],
	bases_bdry: Sequence[jax.Array],
	dbases: Sequence[jax.Array],
	f: jax.Array,
	XW: jax.Array,
	XW_bdry: jax.Array,
) -> jax.Array:
	n_bases = len(bases)
	K = jnp.zeros(shape=(n_bases, n_bases))
	F = jnp.zeros(shape=(n_bases, 1))
	for i in range(n_bases):
		for j in range(i + 1):
			K_ij = bilinear_op(
				u=bases[i],
				v=bases[j],
				du=dbases[i],
				dv=dbases[j],
				u_bdry=bases_bdry[i],
				v_bdry=bases_bdry[j],
				XW=XW,
				XW_bdry=XW_bdry
			)
			K = K.at[i, j].set(K_ij)
		L_i = linear_op(f=f, v=bases[i], XW=XW)
		F = F.at[i, 0].set(L_i)
	K = K + K.T - jnp.diag(jnp.diag(K))
	sol_coeff, _, _, _ = jnp.linalg.lstsq(K, F) # Get solution coefficients
	return sol_coeff


# @jax.jit
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
	# net and dnets shape (xnodes, neurons)
	n_neurons = net.shape[1]
	K = jnp.zeros(shape=(n_neurons, n_neurons))
	F = jnp.zeros(shape=(n_neurons, 1))
	for i in range(n_neurons):
		for j in range(i + 1):
			K_ij = bilinear_op(
				u=net[:, [i]],
				v=net[:, [j]],
				du=dnet[:, [i]],
				dv=dnet[:, [j]],
				u_bdry=net_bdry[:, [i]],
				v_bdry=net_bdry[:, [j]],
				XW=XW,
				XW_bdry=XW_bdry
			)
			K = K.at[i, j].set(K_ij)
		L_i = linear_op(f=f, v=net[:, [i]], XW=XW)
		a_i = bilinear_op(
				u=u,
				v=net[:, [i]],
				du=du,
				dv=dnet[:, [i]],
				u_bdry=u_bdry,
				v_bdry=net_bdry[:, [i]],
				XW=XW,
				XW_bdry=XW_bdry
			)
		F = F.at[i, 0].set(L_i - a_i)
	K = K + K.T - jnp.diag(jnp.diag(K))  # Fill trian-upper entries
	coeff, _, _, _ = jnp.linalg.lstsq(K, F) # Get Galerkin coefficients
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
) -> float:
	# Net with input layer and hidden layer
	net = single_net(params=params, X=X, activation=activation)
	net_bdry = single_net(params=params, X=X_bdry, activation=activation)
	dnet = dsingle_net(X, params, activation).squeeze(axis=-1)
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
	v_nn = net_proj(X=X, params=params, activation=activation, coeff=v_nn_coeff)
	v_nn_bdry = net_proj(X=X_bdry, params=params, activation=activation, coeff=v_nn_coeff)
	dv_nn = dnet_proj(X, params, activation, v_nn_coeff).squeeze(axis=-1)

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
	return -loss, v_nn_coeff


@partial(jax.jit, static_argnums=(0, 11))
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
):
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
	return opt_state, params, -loss, coeff


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
	optimizer = optax.adam(learning_rate=learning_rate)
	key_W, key_b = jax.random.split(key, num=2)
	params = {
		'W': jax.random.normal(shape=(1, neurons), key=key_W),
		'b': jax.random.normal(shape=(neurons, ), key=key_b),
	}
	opt_state = optimizer.init(params)
	for i in range(1, max_epoch + 1):
		opt_state, params, loss, coeff = train_step(
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
		if i % 10 == 0:
			print(f'step {i}, loss: {loss}')
		if jnp.abs(loss) < tol_basis:
			# TODO: Implement a better stopper.
			pass

	# Get the final basis
	net = single_net(X=X, params=params, activation=activation)
	net_bdry = single_net(X=X_bdry, params=params, activation=activation)
	dnet = dsingle_net(X, params, activation).squeeze(axis=-1)
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
	v_nn = net_proj(X=X, params=params, coeff=v_nn_coeff, activation=activation)
	v_nn_bdry = net_proj(X=X_bdry, params=params, activation=activation, coeff=v_nn_coeff)
	dv_nn = dnet_proj(X, params, activation, v_nn_coeff).squeeze(axis=-1)
	v_nn_norm = norm(v=v_nn, dv=dv_nn, v_bdry=v_nn_bdry, XW=XW, XW_bdry=XW_bdry)
	# Basis $\phi^{NN}$
	phi_nn = v_nn / v_nn_norm
	phi_nn_bdry = v_nn_bdry / v_nn_norm
	dphi_nn = dv_nn / v_nn_norm
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
	return phi_nn, phi_nn_bdry, dphi_nn, eta, params, coeff


def adaptive_subspace(
	xbounds: Tuple[float, float],
	source: Callable[[jax.Array], jax.Array],
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
	X_bdry = jnp.array([xa, xb], dtype=float).reshape(-1, 1)
	XW_bdry = jnp.array([2.0, 1.0]).reshape(-1, 1)  # Hardcoded for now
	f_train = source(X_train)

	eta_errors = []  # Remember, for $\eta_i$ we need $\phi_{i+1}^{NN}$
	solution_coeffs = []
	bases_params = []
	bases_coeffs = []
	bases_train = []  # $\phi_i^{NN}(X_train)$
	bases_bdry = []
	dbases_train = []

	# Zero basis
	print(f"Basis: 0")
	u_train = u0(X_train)
	du_train = du0(X_train)
	u_bdry = u0(X_bdry)
	u_norm = norm(v=u_train, dv=du_train, v_bdry=u_bdry, XW=XW_train, XW_bdry=XW_bdry)
	phi_nn_0 = u_train / u_norm if u_norm != 0 else u_train
	phi_nn_0_bdry = u_bdry / u_norm if u_norm != 0 else u_bdry
	dphi_nn_0 = du_train / u_norm if u_norm != 0 else du_train

	bases_train.append(phi_nn_0)
	bases_bdry.append(phi_nn_0_bdry)
	dbases_train.append(dphi_nn_0)

	solution_coeffs.append(jnp.array([1.0]))
	bases_params.append(None)  # There are no NN Weights and Biases for $\phi_0^{NN}$
	bases_coeffs.append(None)  # There are no NN Coefficients for $\phi_0^{NN}$

	# First basis step
	print(f"Basis: 1")
	activation = activations_fn(1)
	neurons = network_widths_fn(1)
	learning_rate = learning_rates_fn(1)
	phi_nn, phi_nn_bdry, dphi_nn, eta, params, coeff = augment_basis(
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
		key=key
	)
	eta_errors.append(eta)
	bases_params.append(params)
	bases_coeffs.append(coeff)
	bases_train.append(phi_nn)
	bases_bdry.append(phi_nn_bdry)
	dbases_train.append(dphi_nn)
	# Basis step loop
	bstep = 2
	while (eta_errors[-1] > tol_solution) and (bstep <= max_bases):
		print(f"Basis: {bstep}")
		u_coeff = galerkin_solve(
			bases_train,
			bases_bdry,
			dbases_train,
			f_train,
			XW_train,
			XW_bdry,
		)
		u_train = solution_proj(u_coeff, bases=bases_train)
		u_bdry = solution_proj(u_coeff, bases=bases_bdry)
		du_train = solution_proj(u_coeff, bases=dbases_train)

		activation = activations_fn(bstep)
		neurons = network_widths_fn(bstep)
		learning_rate = learning_rates_fn(bstep)
		key, _ = jax.random.split(key, num=2)

		phi_nn, phi_nn_bdry, dphi_nn, eta, params, coeff = augment_basis(
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
		solution_coeffs.append(u_coeff)
		eta_errors.append(eta)
		bases_params.append(params)
		bases_coeffs.append(coeff)
		bases_train.append(phi_nn)
		bases_bdry.append(phi_nn_bdry)
		dbases_train.append(dphi_nn)
		bstep += 1

	return (
		eta_errors,
		solution_coeffs,
		bases_params,
		bases_coeffs,
		bases_train,
		bases_bdry,
		dbases_train
	)

# %%
# PDE
xbounds = 0.0, 1.0
def source(X: jax.Array) -> jax.Array:
	f1 = (2 * jnp.pi) ** 2 * jnp.sin(2 * jnp.pi * X)
	f2 = (4 * jnp.pi) ** 2 * jnp.sin(4 * jnp.pi * X)
	f3 = (6 * jnp.pi) ** 2 * jnp.sin(6 * jnp.pi * X)
	return f1 + f2 + f3
u0 = lambda X: jnp.zeros_like(X)
du0 = lambda X: jnp.zeros_like(X)

# NN
n_train = 128
n_val = 1000
N = 5
r = 2
A = 2 * 1e-2
rho = 1.1

beta_fn = lambda i: i
def activations_fn(beta_i):
	def activation(x):
		return jnp.tanh(beta_i * x)
	return activation

network_widths_fn = lambda i: N * r ** (i - 1)
learning_rates_fn = lambda i: A * rho ** (-(i - 1))

max_bases = 8
max_epoch_basis = 100
tol_solution = 1e-9
tol_basis = 1e-4
seed = 42

# %%
(
	eta_errors,
	solution_coeffs,
	bases_params,
	bases_coeffs,
	bases_train,
	bases_bdry,
	dbases_train
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

