# %%
import jax
import jax.numpy as jnp
import numpy as np
import optax

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from flax import nnx

from __future__ import annotations

# %%
class Count(nnx.Variable[nnx.A]):
  pass


class SingleLayer(nnx.Module):
	"""
	Single layer neural network with an activation function
	"""
	def __init__(self, dim_in: int, dim_out: int, activation: Callable, *, rngs: nnx.Rngs):
		key = rngs.params()
		self.count = Count(jnp.array(0))
		initializer = nnx.initializers.uniform(scale=1.0)
		self.w = nnx.Param(initializer(key, (dim_in, dim_out)))
		self.b = nnx.Param(jnp.zeros((dim_out,)))
		self.activation = activation
		self.dim_in, self.dim_out = dim_in, dim_out

	def __call__(self, x):
		self.count.value += 1
		x = x @ self.w + self.b
		x = self.activation(x)
		return x


class DomainNodes:
	"""
	1D domain object
	"""
	def __init__(
		self,
		bounds: tuple[float, float],
		n: int,
	):
		self.bounds = bounds
		self.n = n
		self.values, self.weights = self.gauss_lengendre_quad(bounds, n)
		self.bdry = jnp.array([xa, xb], dtype=float).reshape(-1, 1)
		self.bdry_weights = jnp.array([2.0, 1.0]).reshape(-1, 1)  # Hardcoded for now

	@staticmethod
	def gauss_lengendre_quad(bounds, n):
		# a, b -- 1D domain
    	# ng -- number of nodes for training
		a, b = bounds
		x, w = np.polynomial.legendre.leggauss(deg=n)
		x = 0.5 * (b - a) * x + 0.5 * (b + a)  # Translation from [-1, 1] to [a, b]
		w = 0.5 * (b - a) * w  # Scale quadrature weights
		return jnp.array(x).reshape(-1, 1), jnp.array(w).reshape(-1, 1)


class Function:
	"""
	Function class
	"""
	def __init__(self, f: Callable, df: Callable=None):
		self.f = f
		self.df = df if df is not None else jax.jacobian(f)
		self.vdf = jax.vmap(self.df)

	def __call__(self, X):
		return self.f(X)

	def diff(self, X):
		try:
			return self.vdf(X).squeeze(axis=-1)  # Try vectorized form
		except:
			return self.df(X)

	def __repr__(self) -> str:
		return f"Function {self.f} and its derivative {self.df})"

	@classmethod
	def linear_combination(
		cls,
		bases: Sequence[Function],
		coeffs: Sequence[float],
	) -> Function:

		if len(coeffs) != len(bases):
			raise ValueError("coeffs and bases must have the same length")

		def combined_f(x):
			return jnp.sum(c * f(x) for c, f in zip(coeffs.flatten(), bases))

		def combined_df(x):
			return jnp.sum(c * f.diff(x) for c, f in zip(coeffs.flatten(), bases))

		return cls(combined_f, combined_df)


@dataclass
class PDEvar:
	"""
	Variational PDE
	"""
	bilinear: Callable
	linear: Callable

	def norm(self, v: Function, domain: DomainNodes) -> float:
		"""
		Norm |||v|||
		"""
		a = self.bilinear(u=v, v=v, domain=domain)
		return jnp.sqrt(a)

	def residual(self, u: Function, v: Function, domain: DomainNodes) -> float:
		"""
		Residual L(v) - a(u, v)
		"""
		L_v = self.linear(v=v, domain=domain)
		a_uv = self.bilinear(u=u, v=v, domain=domain)
		return L_v - a_uv

	def error_eta(self, u: Function, v: Function, domain: DomainNodes) -> float:
		r"""Error
		$$ \eta(u, v) = <r(u), v> / |||v||| = (L(v) - a(u, v)) / |||v||| $$
		"""
		norm_v = self.norm(v=v, domain=domain)
		return self.residual(u=u, v=v, domain=domain) / norm_v


def galerkin_solve(pde: PDEvar, domain: DomainNodes, bases: list[Function]) -> jax.Array:
	n_bases = len(bases)
	K = jnp.zeros(shape=(n_bases, n_bases))
	F = jnp.zeros(shape=(n_bases, 1))
	for i in range(n_bases):
		for j in range(i + 1):
			K = K.at[i, j].set(pde.bilinear(u=bases[i], v=bases[j], domain=domain))
		F = F.at[i, 0].set(pde.linear(v=bases[i], domain=domain))
	K = K + K.T - jnp.diag(jnp.diag(K))  # Fill trian-upper entries
	coeffs, _, _, _ = jnp.linalg.lstsq(K, F) # Get Galerkin coefficients
	return coeffs


def galerkin_lsq(
		pde: PDEvar,
		u_prev: Function,
		model: SingleLayer,
		domain: DomainNodes
	) -> jax.Array:
	neurons = model.dim_out
	# model_dx = lambda x: jax.vmap(jax.jacobian(model))(x).squeeze(axis=-1)
	K = np.zeros(shape=(neurons, neurons))
	F = np.zeros(shape=(neurons, 1))
	sigmas = [Function(lambda x: model(x)[:, [i]]) for i in range(neurons)]
	for i in range(neurons):
		for j in range(i + 1):
			K = K.at[i, j].set(pde.bilinear(u=sigmas[i], v=sigmas[j], domain=domain))
		F = F.at[i, 0].set(
			pde.linear(v=sigmas[i], domain=domain)
			- pde.bilinear(u=u_prev, v=sigmas[i])
		)
	K = K + K.T - jnp.diag(jnp.diag(K))  # Fill trian-upper entries
	coeffs, _, _, _ = jnp.linalg.lstsq(K, F) # Get Galerkin coefficients
	pass


def augment_basis(
	pde: PDEvar,
	domain: DomainNodes,
	u_prev: Function,
	neurons: int,
	activation: Callable,
	learning_rate: float,
	rng: nnx.Rngs,
	max_epoch: int,
) -> tuple[Function, float]:

	model = SingleLayer(
		dim_in=1,
		dim_out=neurons,
		activation=activation,
		rngs=rng
	)
	optimizer = nnx.Optimizer(model, optax.sgd(learning_rate))
	losses = []
	loss_prev = 1e6

	@nnx.jit
	def train_step(model, optimizer):
		def loss_fn(model, domain):
			c = galerkin_lsq(u=u_prev, pde=pde, model=model, domain=domain)
			phi = Function(lambda x: model(x) @ c)
			loss = pde.error_eta(u=u_prev, v=phi, domain=domain)
			return loss, phi

		grads_fn = nnx.value_and_grad(loss_fn, has_aux=True, argnums=(0,))
		(loss, phi), grads = grads_fn(model, domain)
		optimizer.update(grads)
		return loss, phi

	# Training process
	for _ in range(max_epoch):
		loss, phi = train_step(model, optimizer)
		losses.append(np.asarray(loss))
		loss_relative = (loss - loss_prev) / loss_relative  # Stop criteria, TODO: Be careful with the pick of Learning Rate
		if loss_relative < tol_basis:
			break
		else:
			loss_prev = loss
	bases_losses.append(np.asarray(losses))  # Append history of each training

	# Return normalized $\phi^{NN}$
	phi_norm = pde.norm(v=phi, domain=domain)
	phi_unit = Function(lambda x: phi(x) / phi_norm)
	eta = pde.error_eta(u=u_prev, v=phi_unit, domain=domain)
	return phi_unit, eta


# --------------------------------------------------------------
# Data
seed = 42
key = jax.random.key(seed)

# NN
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

# PDE
xa, xb = 0.0, 1.0
n_domain = 32
n_domain_val = 20
u0 = Function(f=lambda x: 0.0, df=lambda x: 0.0)

def poisson_linear(v: Function, domain: DomainNodes) -> float:
	X, X_weights = domain.values, domain.weights
	L_v = jnp.sum(X_weights * v(X) * 0.0)  # TODO
	return L_v

def poisson_bilinear(u: Function, v: Function, domain: DomainNodes) -> float:
	X, X_weights = domain.values, domain.weights
	a_uv = jnp.sum(X_weights * u.diff(X) * v.diff(X))
	return a_uv

domain = DomainNodes(bounds=(xa, xb), n=n_domain)
domain_val = DomainNodes(bounds=(xa, xb), n=n_domain_val)
pde = PDEvar(bilinear=poisson_bilinear, linear=poisson_linear)

# ----------------------
# Algorithm starts here
# ----------------------
# Subspace construction
u_list = [u0]
u_coeffs = [jnp.array([1.0])]
norm_u0 = pde.norm(v=u0, domain=domain)
bases = [Function(f=lambda x: u0(x) / norm_u0)]
bases_losses = []
eta_errors = []

# %%
key, subkey = jax.random.split(key)
phi_nn, eta = augment_basis(  # TODO
	pde=pde,
	domain=domain,
	u_prev=u0,
	neurons=network_widths(i=0),
	activation = lambda x: activations(i=0, x=x),
	learning_rate=learning_rates(0),
	rng=nnx.Rngs({'params': subkey}),
	max_epoch=max_epoch
)
bases.append(phi_nn)
eta_errors.append(eta)

bstep = 1
while (eta_errors[-1] > tol_solution) and (bstep <= max_basis):
	key, subkey = jax.random.split(key)
	coeffs = galerkin_solve(pde, domain)
	u = Function.linear_combination(bases, coeffs)
	phi_nn, eta = augment_basis(
		pde=pde,
		domain=domain,
		u_prev=u,
		neurons=network_widths(i=bstep),
		activation = lambda x: activations(i=bstep, x=x),
		learning_rate=learning_rates(bstep),
		rng=nnx.Rngs({'params': subkey}),
		max_epoch=max_epoch
	)
	u_list.append(u)
	bases.append(phi_nn)
	eta_errors.append(eta)
	bstep += 1





# ----------------------
# Playground
# ----------------------
rng = nnx.Rngs({'params': key})
model = SingleLayer(
	dim_in=1,
	dim_out=4,
	activation=jnp.tanh,
	rngs=rng
)