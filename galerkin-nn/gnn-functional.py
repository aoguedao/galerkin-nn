# %%
import jax
import jax.numpy as jnp
import numpy as np
import optax


# Neural Network
def single_net(X: jax.Array, params: optax.Params) -> jnp.ndarray:
  X = jnp.dot(X, params["W"]) + params["b"]
  X = jnp.tanh(X) # TODO
  return X


def net_proj(X: jax.Array, params: optax.Params, coeff: jax.Array):
	"""
	Linear combination of the neural network
	"""
	# assert X.shape[1] == coeff.shape[1]
	return jnp.dot(single_net(X=X, params=params), coeff)


# PDE
def gauss_lengendre_quad(bounds: tuple, n: int) -> tuple[jax.Array, jax.Array]:
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
	eps = 1  # TODO
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


# Galerkin
def galerkin_solve():
	pass


def galerkin_lsq(
	u,
	du,
	u_bdry,
	net,
	dnet,
	net_bdry,
	f,
	XW,
	XW_bdry,
):
	# net, dnet shape (xnodes, neurons)
	neurons = net.shape[1]
	K = jnp.zeros(shape=(neurons, neurons))
	F = jnp.zeros(shape=(neurons, 1))
	for i in range(neurons):
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
	coeffs, _, _, _ = jnp.linalg.lstsq(K, F) # Get Galerkin coefficients
	return coeffs

def augment_basis():
	pass


# %%
# Data
seed = 42
key = jax.random.key(seed)

N = 5
r = 1
A = 0.01
rho = 1.1
n_domain = 32
max_basis = 3
network_widths = lambda i: N * r ** i
activations = lambda i, x: jnp.tanh((i + 1) * x)
learning_rates = lambda i: A * rho ** (-i)
max_epoch = 1_000
tol_solution = 1e-6
tol_basis = 1e-6

# Domain
xa, xb = 0.0, 1.0
n_train = 32
n_val = 20
X_train, XW_train = gauss_lengendre_quad((xa, xb), n_train)
X_val, XW_val = gauss_lengendre_quad((xa, xb), n_val)
X_bdry = jnp.array([xa, xb], dtype=float).reshape(-1, 1)
XW_bdry = jnp.array([2.0, 1.0]).reshape(-1, 1)  # Hardcoded for now

# PDE
def source(X: jax.Array) -> jax.Array:
	f1 = (2 * jnp.pi) ** 2 * jnp.sin(2 * jnp.pi * X)
	f2 = (4 * jnp.pi) ** 2 * jnp.sin(4 * jnp.pi * X)
	f3 = (6 * jnp.pi) ** 2 * jnp.sin(6 * jnp.pi * X)
	return f1 + f2 + f3

u0 = lambda x: jnp.zeros_like(x)
du0 = lambda x: jnp.zeros_like(x)

# %%
# ----------------------
# Algorithm starts here
# ----------------------
# Subspace construction

f_train = source(X_train)
u_train = u0(X_train)
du_train = du0(X_train)
u_bdry = u0(X_bdry)
u_norm = norm(v=u_train, dv=du_train, v_bdry=u_bdry, XW=XW_train, XW_bdry=X_bdry)

if u_norm == 0:
	bases = [u_train]
else:
	bases = [u_train / u_norm]
u_coeffs = [jnp.array([1.0])]

# --------------------------------------
# WORK IN PROGRESS
# --------------------------------------
# basis_params = []
# phi_params, phi_coeff, eta_error = augment_basis(u_prev_train)  # TODO
# eta_errors = [eta_error]

# bstep = 1
# while (eta_errors[-1] > tol_solution) and (bstep <= max_basis):
# 	u_coeffs = galerkin_solve() # TODO
# 	u = u_coeffs @ bases  # TODO
# 	phi_nn, eta = augment_basis(u)
# 	u_list.append(u)
# 	bases.append(phi_nn)
# 	eta_errors.append(eta)
# 	bstep += 1
# --------------------------------------







# %%
# --------------------------------------
# AUGMENT BASIS PLAYGROUND
# --------------------------------------
neurons = 4
key_W, key_b = jax.random.split(key, num=2)
params_init = {
    'W': jax.random.normal(shape=(1, neurons), key=key_W),
    'b': jax.random.normal(shape=(neurons, ), key=key_b),
}

# optimizer = optax.adam(learning_rate=1e-2)
# params = params_init

# net_train = single_net(params=params, X=X_train)
# net_bdry = single_net(params=params, X=X_bdry)
# dnet_train = jax.vmap(jax.jacobian(single_net, argnums=0), in_axes=(0, None))(X_train, params).squeeze(axis=-1)

# v_nn_coeff = galerkin_lsq(
# 	u=u_train,
# 	du=du_train,
# 	u_bdry=u_bdry,
# 	net=net_train,
# 	dnet=dnet_train,
# 	net_bdry=net_bdry,
# 	f=f_train,
# 	XW=XW_train,
# 	XW_bdry=XW_bdry,
# )
# v_nn_train = net_proj(X=X_train, params=params, coeff=v_nn_coeff)
# v_nn_bdry = net_proj(X=X_bdry, params=params, coeff=v_nn_coeff)
# dv_nn_train = jnp.dot(dnet_train, v_nn_coeff)

# error = error_eta(
# 	u=u_train,
# 	du=du_train,
# 	u_bdry=u_bdry,
# 	v=v_nn_train,
# 	dv=dv_nn_train,
# 	v_bdry=v_nn_bdry,
# 	f=f_train,
# 	XW=XW_train,
# 	XW_bdry=XW_bdry
# )

def loss_fn(
	params: optax.Params,
	u: jax.Array,
	du: jax.Array,
	u_bdry: jax.Array,
	f: jax.Array,
	X: jax.Array,
	XW: jax.Array,
	XW_bdry: jax.Array
):
	# Net with input layer and hidden layer
	net = single_net(params=params, X=X)
	net_bdry = single_net(params=params, X=X_bdry)
	dnet = (
		jax.vmap(
			jax.jacobian(single_net, argnums=0),
			in_axes=(0, None)
		)(X, params)
		.squeeze(axis=-1)
	)
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
	v_nn = net_proj(X=X, params=params, coeff=v_nn_coeff)
	v_nn_bdry = net_proj(X=X_bdry, params=params, coeff=v_nn_coeff)
	dv_nn = jnp.dot(dnet, v_nn_coeff)
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

@jax.jit
def train_step(
	params: optax.Params,
	opt_state: tuple,
	u: jax.Array,
	du: jax.Array,
	u_bdry: jax.Array,
	f: jax.Array,
	X: jax.Array,
	XW: jax.Array,
	XW_bdry: jax.Array,
):
	# optimizer is a global variable, I am not sure if this will impact
    (loss, coeff), grads = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)(
		params,
		u,
		du,
		u_bdry,
		f,
		X,
		XW,
		XW_bdry
	)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, coeff

params = params_init
optimizer = optax.adam(learning_rate=1e-2)
opt_state = optimizer.init(params)
for i in range(max_epoch):
	params, opt_state, loss_value, coeff = train_step(
		params=params,
		opt_state=opt_state,
		u=u_train,
		du=du_train,
		u_bdry=u_bdry,
		f=f_train,
		X=X_train,
		XW=XW_train,
		XW_bdry=XW_bdry
	)
	if i % 100 == 0:
		print(f'step {i}, loss: {loss_value}')
	break



















# %%
