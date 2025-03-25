# %%
import jax
import jax.numpy as jnp
import numpy as np
import optax

from collections.abc import Callable
from jax.typing import ArrayLike
from flax import nnx

import utils


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


class GalerkinNN1D():
	def __init__(
		self,
		a:float,
		b: float,
		data: Callable,
		bilinear: Callable,
		u_initial: Callable,
		d_u_initial: Callable,
		n_x: int,
		n_x_val: int,
		network_widths: ArrayLike,
		activations: ArrayLike,
		learning_rates: ArrayLike,
		key,
		max_basis: int=50,
		solution_tol: float=1e-6,
		basis_tol: float=1e-6,
		max_epoch: int=1_000
	):
		# PDE formulation
		self.a, self.b = a, b
		self.data = data
		self.bilinear = bilinear
		self.u0 = u_initial
		self.d_u0 = d_u_initial
		self.n_x = n_x
		self.n_x_val = n_x_val
		self.X, self.X_weights = utils.gauss_lengendre_quad(self.a, self.b, self.n_x)
		self.X_val, self.X_val_weights = utils.gauss_lengendre_quad(self.a, self.b, self.n_x_val)
		self.X_bdry = jnp.array([self.a, self.b], dtype=float).reshape(-1, 1)
		self.X_bdry_weights = jnp.array([2.0, 1.0]).reshape(-1, 1)  # Hardcoded for now
		# Network Architecture
		self.network_widths = network_widths
		self.activations = activations
		self.learning_rates= learning_rates
		self.max_basis = max_basis
		# Utils
		self.key = key
		self.keys = jax.random.split(self.key, self.max_basis)
		self.rngs = [nnx.Rngs({'params': subkey}) for subkey in self.keys]
		self.solution_tol = solution_tol
		self.basis_tol = basis_tol
		self.max_epoch = max_epoch
		self.basis = []
		self.u_list = [u_initial]
		self.error_eta_list = []
		self.basis_losses = []
		self.models = []
		self.i = 0


	def subspace_construction(self):
		norm_u0 = self.norm(self.u0, self.d_u0)
		self.basis.append(lambda x: self.u0(x) / norm_u0)  # Initial approximation

		phi_nn, error_eta = self.augment_basis(self.u0)  # Augment basis
		self.basis.append(phi_nn)
		self.error_eta_list.append(error_eta)

		self.i = 1
		while (self.error_eta_list[-1] > self.tol) and (self.i <= self.max_basis):
			i = self.i
			# Approx. solution
			u = self.galerkin_solve(self.basis, self.bilinear, self.data)
			# Augment basis
			phi_nn, error_eta = self.augment_basis(self.u_list[i])
			# Append to lists
			self.u_list.append(u)
			self.basis.append(phi_nn)
			self.error_eta_list.append(error_eta)
			self.i += 1


	def augment_basis(self, u_prev):
		activation_i = self.activations[self.i]
		neurons_i = self.network_widths[self.i]
		lr_i = self.learning_rates[self.i]
		rngs_i = self.rngs[self.i]
		model = SingleLayer(
			dim_in=1,
			dim_out=neurons_i,
			activation=activation_i,
			rngs=rngs_i
		)
		optimizer = nnx.Optimizer(model, optax.sgd(lr_i))
		d_u_prev = jax.grad(u_prev)
		losses = []
		loss_prev = 1e6

		@nnx.jit
		def train_step(model, optimizer):
			def loss_fn(model, X, X_weights):
				residual_u = lambda v, d_v: self.residual(u_prev, v, d_u_prev, d_v, X, X_weights)
				c = self.galerkin_lsq(model, self.bilinear, residual_u)
				phi = lambda x: model(x) @ c
				d_phi = jax.grad(phi)
				loss = self.error_eta(u_prev, phi, d_u_prev, d_phi, self.X, self.X_weights)
				return loss, phi

			loss, grads = nnx.value_and_grad(loss_fn)(model, self.X, self.X_weights)
			optimizer.update(grads)
			return loss, phi

		# Training process
		for _ in range(self.max_epoch):
			loss, phi = train_step(model, optimizer)
			losses.append(np.asarray(loss))
			loss_relative = (loss - loss_prev) / loss_relative  # Stop criteria, TODO: Be careful with the pick of Learning Rate
			if loss_relative < self.basis_tol:
				break
			else:
				loss_prev = loss
		self.basis_losses.append(np.asarray(losses))

		# Return normalized $\phi^{NN}$
		d_phi = jax.grad(phi)
		phi_unit = lambda x: phi(x) / self.norm(phi, d_phi)
		d_phi_unit = jax.grad(phi_unit)
		eta = self.error_eta(u_prev, phi_unit, d_u_prev, d_phi_unit)
		return phi_unit, eta


	def galerkin_solve(self, bases, bilinear, data):
		n_bases = len(bases)
		K = np.zeros(shape=(n_bases, n_bases))
		F = np.zeros(shape=(n_bases, 1))
		for i in range(n_bases):
			phi_i = bases[i]
			d_phi_i = jax.grad(phi_i)
			for j in range(i + 1):
				phi_j = bases[j]
				d_phi_j = jax.grad(phi_i)
				K[i, j] = bilinear(
					u=phi_i,
					v=phi_j,
					d_u=d_phi_i,
					d_v=d_phi_j,
					X=self.X,
					X_weights=self.X_weights,
					X_bdry=self.X_bdry,
					X_bdry_weights=self.X_bdry_weights
				)
			F[i, 0] = data(
				u=phi_i,
				d_u=d_phi_i,
				X=self.X,
				X_weights=self.X_weights,
				X_bdry=self.X_bdry,
				X_bdry_weights=self.X_bdry_weights
			)
		K += K.T - np.diag(np.diag(K))  # Fill upper-triang entries
		# Get Galerkin coefficients
		c, _, _, _ = jnp.linalg.lstsq(K, F)
		u = lambda x: jnp.sum(basis(x) @ c[i, 0] for (basis, i) in enumerate(bases))
		return u


	def galerkin_lsq(self, model, bilinear, residual_u):
		neurons = model.dim_out
		model_dx = jax.vmap(jax.jacobian(model))
		K = np.zeros(shape=(neurons, neurons))
		F = np.zeros(shape=(neurons, 1))
		for i in range(neurons):
			sigma_i = lambda x: model(x)[:, [i]]
			d_sigma_i = lambda x: model_dx(x).squeeze()[:, [i]]
			for j in range(i + 1):
				sigma_j = lambda x: model(x)[:, [j]]
				d_sigma_j = lambda x: model_dx(x).squeeze()[:, [j]]
				K[i, j] = bilinear(
					u=sigma_i,
					v=sigma_j,
					d_u=d_sigma_i,
					d_v=d_sigma_j,
					X=self.X,
					X_weights=self.X_weights,
					X_bdry=self.X_bdry,
					X_bdry_weights=self.X_bdry_weights
				)
			F[i, 0] = residual_u(
				v=sigma_i,
				d_v=d_sigma_i,
				X=self.X,
				X_weights=self.X_weights,
				X_bdry=self.X_bdry,
				X_bdry_weights=self.X_bdry_weights
			)
		K += K.T - np.diag(np.diag(K))  # Fill upper-triang entries
		c, _, _, _ = jnp.linalg.lstsq(K, F)
		return c


	def residual(
		self,
		u: Callable,
		v: Callable,
		d_u: Callable,
		d_v: Callable,
		X: ArrayLike,
		X_weights: ArrayLike
	):
		"""Residual L(v) - a(u, v)

		Parameters
		----------
		u : Callable
			_description_
		v : Callable
			_description_
		d_u : Callable
			_description_
		d_v : Callable
			_description_
		X : ArrayLike
			_description_
		X_weights : ArrayLike
			_description_

		Returns
		-------
		_type_
				_description_
		"""
		L = self.data(
			u=u,
			d_u=d_u,
			X=X,
			X_weights=X_weights,
			X_bdry=self.X_bdry,
			X_bdry_weights=self.X_bdry_weights
		)
		a_u = self.bilinear(
			u=u,
			v=v,
			d_u=d_u,
			d_v=d_v,
			X=X,
			X_weights=X_weights,
			X_bdry=self.X_bdry,
			X_bdry_weights=self.X_bdry_weights
		)
		return L - a_u


	def norm(
		self,
		v: Callable,
		d_v: Callable,
		X: ArrayLike=None,
		X_weights:ArrayLike=None,
	):
		"""Norm |||v|||

		Parameters
		----------
		v : Callable
				_description_
		d_v : Callable
				_description_
		X : ArrayLike, optional
				_description_, by default None
		X_weights : ArrayLike, optional
				_description_, by default None

		Returns
		-------
		_type_
				_description_
		"""
		if None in [X, X_weights]:
			X, X_weights = self.X, self.X_weights
		a = self.bilinear(
			u=v,
			v=v,
			d_u=d_v,
			d_v=d_v,
			X=X,
			X_weights=X_weights,
			X_bdry=self.X_bdry,
			X_bdry_weights=self.X_bdry_weights,
		)
		return jnp.sqrt(a)


	def error_eta(
			self,
			u: Callable,
			v: Callable,
			d_u: Callable,
			d_v: Callable,
			X: ArrayLike,
			X_weights: ArrayLike
	):
		r"""Error

		$$ \eta(u, v) = <r(u), v> / |||v||| = (L(v) - a(u, v)) / |||v||| $$

		Parameters
		----------
		u : Callable
			_description_
		phi : Callable
			_description_
		d_u : Callable
			_description_
		d_phi : Callable
			_description_
		X : ArrayLike
			_description_
		X_weights : ArrayLike
			_description_

		Returns
		-------
		_type_
			_description_
		"""
		phi_norm = self.norm(v, d_v)
		return self.residual(u, v, d_u, d_v, X, X_weights) / phi_norm


# %%
seed = 42
key = jax.random.key(seed)
a, b = 0.0, 1.0

def data(
	u,
	d_u,
	X,
	X_weights,
	X_bdry,
	X_bdry_weights
):
	return jnp.sum(X_weights * u(X) * 0.0)  # Hardcoded for now


def bilinear(
	u,
	v,
	d_u,
	d_v,
	X,
	X_weights,
	X_bdry,
	X_bdry_weights
):
	return jnp.sum(X_weights * d_u(X) * d_v(X))


u_initial = lambda x: 0.0
d_u_initial = lambda x: 0.0

N = 5
r = 1
A = 0.01
rho = 1.1
n_x = 100
n_x_val = 20
max_basis = 3
network_widths = [N * r ** i for i in range(max_basis)]
activations = [lambda x: jnp.tanh((i + 1) * x) for i in range(max_basis)]
learning_rates = [A * rho ** (-i) for i in range(max_basis)]
solution_tol = 1e-6
basis_tol = 1e-6

# %%
gnn = GalerkinNN1D(
	a=a,
	b=b,
	data=data,
	bilinear=bilinear,
	u_initial=u_initial,
	d_u_initial=d_u_initial,
	n_x=n_x,
	n_x_val=n_x_val,
	network_widths=network_widths,
	activations=activations,
	learning_rates=learning_rates,
	key=key,
	max_basis=max_basis,
	solution_tol=solution_tol,
	basis_tol=basis_tol,
)

gnn.subspace_construction()
# %%
