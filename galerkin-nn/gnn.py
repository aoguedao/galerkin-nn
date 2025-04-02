import jax
import jax.numpy as jnp
import numpy as np
import optax

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from jax.typing import ArrayLike
from flax import nnx

from __future__ import annotations

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
	def gaus_legendre_quad(bounds, n):
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
		if df is None:
			self.df = jax.grad(f)
		else:
			self.df = df

	def __call__(self, *args, **kwds):
		return self.f(*args, **kwds)

	def diff(self, *args, **kwds):
		return self.df(*args, **kwds)

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
	bilinear: Callable
	linear: Callable

	def norm(self, v: Function, Omega: DomainNodes) -> float:
		"""
		Norm |||v|||
		"""
		a = self.bilinear(u=v, v=v, Omega=Omega)
		return jnp.sqrt(a)

	def residual(self, u: Function, v: Function, Omega: DomainNodes) -> float:
		"""
		Residual L(v) - a(u, v)
		"""
		L_v = self.linear(v=v, Omega=Omega)
		a_uv = self.bilinear(u=u, v=v, Omega=Omega)
		return L_v - a_uv

	def error_eta(self, u: Function, v: Function, Omega: DomainNodes) -> float:
		r"""Error
		$$ \eta(u, v) = <r(u), v> / |||v||| = (L(v) - a(u, v)) / |||v||| $$
		"""
		norm_v = self.norm(v=v, Omega=Omega)
		return self.residual(u=u, v=v, Omega=Omega) / norm_v


def galerkin_solve(pde: PDEvar, domain: DomainNodes, bases: list[Function]):
	n_bases = len(bases)
	K = jnp.zeros(shape=(n_bases, n_bases))
	F = jnp.zeros(shape=(n_bases, 1))
	for i in range(n_bases):
		for j in range(i + 1):
			K = K.at[i, j].set(pde.bilinear(u=bases[i], v=bases[j], domain=domain))
		F = F.at[i, 0].set(pde.linear(v=bases[i], domain=domain))
	K = K + K.T - jnp.diag(jnp.diag(K))  # Fill upper-triang entries
	# Get Galerkin coefficients
	coeffs, _, _, _ = jnp.linalg.lstsq(K, F)
	# u = lambda x: jnp.sum(basis(x) @ c[i, 0] for (basis, i) in enumerate(bases))
	return coeffs


def augment_basis():
	pass



# Data
seed = 42
key = jax.random.key(seed)

# NN
N = 5
r = 1
A = 0.01
rho = 1.1
n_x = 100
n_x_val = 20
max_basis = 3
network_widths = lambda i: N * r ** i
activations = lambda i, x: jnp.tanh((i + 1) * x)
learning_rates = lambda i: A * rho ** (-i)
tol_solution = 1e-6
tol_basis = 1e-6

# PDE
xa, xb = 0.0, 1.0
n_domain = 32
u0 = Function(f=lambda x: 0.0, df=lambda x: 0.0)

def poisson_linear(v: Function, Omega: DomainNodes) -> float:
	X, X_weights = Omega.values, Omega.weights
	L_v = jnp.sum(X_weights * v(X) * 0.0)  # TODO
	return L_v

def poisson_bilinear(u: Function, v: Function, Omega: DomainNodes) -> float:
	X, X_weights = Omega.values, Omega.weights
	a_uv = jnp.sum(X_weights * u.diff(X) * v.diff(X))
	return a_uv

domain = DomainNodes(bounds=(xa, xb), n=n_domain)
pde = PDEvar(bililnear=poisson_bilinear, linear=poisson_linear)


# Subspace construction
u_list = [u0]
u_coeffs = [jnp.array([1.0])]
norm_u0 = pde.norm(v=u0, Omega=domain)
bases = [Function(f=lambda x: u0(x) / norm_u0)]
eta_errors = []

phi_nn, eta = augment_basis(u0)  # Augment basis
bases.append(phi_nn)
eta_errors.append(eta)

i = 1
while (eta_errors[-1] > tol_solution) and (i <= max_basis):
	coeffs = galerkin_solve(pde, domain)  # Approx. solution
	u = Function.linear_combination(bases, )
	phi_nn, eta = augment_basis(u)
	u_list.append(u)
	bases.append(phi_nn)
	eta_errors.append(eta)
	i += 1