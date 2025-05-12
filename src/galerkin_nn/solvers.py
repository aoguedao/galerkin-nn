import jax
import jax.numpy as jnp
import optax

from functools import partial
from typing import Callable, Sequence, Tuple

from .domain import Quadrature
from .pde import FunctionState

jax.config.update("jax_enable_x64", True)


class GalerkinNN1D:

  def __init__(
    self,
    pde,
  ):
    self.pde = pde

  def fit(
    self,
    u0: Callable[[jax.Array], jax.Array],
    du0: Callable[[jax.Array], jax.Array],
    max_bases: int,
    net_fns: Sequence[Callable],
    init_net_params: Sequence[optax.Params],
    optimizers: Sequence[optax.GradientTransformation],
    quad_name: str = "gauss-legendre",
    num_train: int = 512,
    num_val: int = 1024,
    max_epoch_per_basis: int = 5_000,
    tol_solution: float = 1e-8,
    tol_basis: float = 1e-6,
  ):

    self.u0 = u0
    self.du0 = du0
    self.max_bases = max_bases
    self.net_fns = net_fns
    self.init_net_params = init_net_params
    self.optimizers = optimizers
    self.quad_name = quad_name
    self.num_train = num_train
    self.num_val = num_val
    self.max_epoch_per_basis = max_epoch_per_basis
    self.tol_solution = tol_solution
    self.tol_basis = tol_basis

    self.eta_errors: Sequence[float] = []
    self.basis_params: Sequence[optax.Params] =[]
    self.basis_coeffs: Sequence[jax.Array] = []
    self.basis_fns: Sequence[Callable] = []
    self.dbasis_fns: Sequence[Callable] = []
    self.bases_train : Sequence[FunctionState] = []
    self.solution_coeffs: Sequence[jax.Array] = []

    _source = self.pde.source()
    _galerkin_solve = self.galerkin_solve()

    # Basis step loop
    basis_step = 0
    quad_train = self.pde.domain.quadrature(
      degree=self.num_train,
      name=self.quad_name
    )
    f_train = _source(X=quad_train.interior)
    u_train = FunctionState(
      interior=self.u0(X=quad_train.interior),
      grad_interior=self.du0(X=quad_train.interior),
      boundary=self.u0(X=quad_train.boundary)
    )

    error_indicator = 1
    while (error_indicator > self.tol_solution) and (basis_step < self.max_bases):
      print(f"Basis Step: {basis_step + 1}")

      net_fn = self.net_fns[basis_step]
      init_net_params = self.init_net_params[basis_step]
      optimizer = self.optimizers[basis_step]

      eta, params, coeff, basis_fn, dbasis_fn, basis_train = self.augment_basis(
        domain_quad=quad_train,
        u=u_train,
        f=f_train,
        net_fn=net_fn,
        init_params=init_net_params,
        optimizer=optimizer,
      )

      self.eta_errors.append(eta)
      self.basis_params.append(params)
      self.basis_coeffs.append(coeff)
      self.basis_fns.append(basis_fn)
      self.dbasis_fns.append(dbasis_fn)
      self.bases_train.append(basis_train)

      bases_train_int = jnp.stack(
        [basis.interior for basis in self.bases_train],
        axis=1
      )
      bases_train_bdry = jnp.stack(
        [basis.boundary for basis in self.bases_train],
        axis=1
      )
      dbases_train_int = jnp.stack(
        [basis.grad_interior for basis in self.bases_train],
        axis=1
      )

      # Solution coefficients
      sol_coeff, stiffness = _galerkin_solve(
        bases=bases_train_int,
        bases_bdry=bases_train_bdry,
        dbases=dbases_train_int,
        f=f_train,
        XW=quad_train.interior_weights,
        XW_bdry=quad_train.boundary_weights,
      )
      self.solution_coeffs.append(sol_coeff)
      u_train = FunctionState(
        interior=jnp.dot(bases_train_int, sol_coeff),
        grad_interior=jnp.dot(dbases_train_int, sol_coeff),
        boundary=jnp.dot(bases_train_bdry, sol_coeff)
      )
      print(f"Solution coeffs: {sol_coeff}")
      error_indicator = jnp.abs(eta)
      basis_step += 1

  def augment_basis(
    self,
    domain_quad: Quadrature,
    u: FunctionState,
    f: jax.Array,
    net_fn: Callable[[jax.Array, optax.Params], jax.Array],
    init_params: optax.Params,
    optimizer: optax.GradientTransformation,
  ):
    # Gradient neural network
    dnet_vmap = jax.vmap(jax.jacobian(net_fn, argnums=0), in_axes=(0, None))
    def dnet_fn(X: jax.Array, params: optax.Params):
      return dnet_vmap(X, params).squeeze()

    # Jitabble loss function that returns coefficients as well
    _loss_fn = self.loss_fn(net_fn=net_fn, dnet_fn=dnet_fn)
    loss_coeff_and_gradloss = jax.jit(
      jax.value_and_grad(
        _loss_fn,
        argnums=0,
        has_aux=True
      )
    )
    # Training
    params = init_params
    opt_state = optimizer.init(params)
    loss_prev = 1
    (loss, net_coeff), grads = loss_coeff_and_gradloss(params, domain_quad, u, f)
    # print(f"Initial loss: {loss:.4e}, grad {grads}")
    for i in range(self.max_epoch_per_basis):
      (loss, net_coeff), grads = loss_coeff_and_gradloss(params, domain_quad, u, f)
      updates, opt_state = optimizer.update(grads, opt_state, params)
      params = optax.apply_updates(params, updates)
      if i % (self.max_epoch_per_basis // 10) == 0:
        print(f"step {i}, loss: {- float(loss)}, grad norm: {optax.global_norm(grads):.4e}")
      if jnp.abs((loss - loss_prev) / loss_prev) < self.tol_basis:
        break
      else:
        loss_prev = loss

    # Basis functions
    def basis_fn(X: jax.Array, params, coeff):
      net = net_fn(X, params)
      return jnp.dot(net, coeff)

    def dbasis_fn(X, params, coeff):
      dnet = dnet_fn(X, params)
      return jnp.dot(dnet, coeff)

    _basis_fn = partial(basis_fn, params=params, coeff=net_coeff)
    _dbasis_fn = partial(dbasis_fn, params=params, coeff=net_coeff)

    basis = FunctionState(
      interior=_basis_fn(domain_quad.interior[:, jnp.newaxis]),
      grad_interior=_dbasis_fn(domain_quad.interior[:, jnp.newaxis]),
      boundary=_basis_fn(domain_quad.boundary[:, jnp.newaxis]),
    )

    # Compute error
    _error_eta = self.pde.error_eta()
    eta = _error_eta(
      u=u.interior,
      du=u.grad_interior,
      u_bdry=u.boundary,
      v=basis.interior,
      dv=basis.grad_interior,
      v_bdry=basis.boundary,
      f=f,
      XW=domain_quad.interior_weights,
      XW_bdry=domain_quad.boundary_weights
    )

    return eta, params, net_coeff, _basis_fn, _dbasis_fn, basis

  def solution_fn(self):
    _basis_fns = self.basis_fns
    _coeff = self.solution_coeffs[-1]
    def _solution_fn(X: jax.Array) -> jax.Array:
      bases = jnp.stack(
        [
          basis_fn(X) for basis_fn in _basis_fns
        ],
        axis=1
      )
      return jnp.dot(bases, _coeff)
    return _solution_fn

  def galerkin_solve(self) -> Callable:
    _linear_operator = self.pde.linear_operator()
    _bilinear_form = self.pde.bilinear_form()
    def _galerkin_solve(
      bases: jax.Array,
      bases_bdry: jax.Array,
      dbases: jax.Array,
      f: jax.Array,
      XW: jax.Array,
      XW_bdry: jax.Array,
    ) -> jax.Array:

      F = jax.vmap(lambda v : _linear_operator(f=f, v=v, XW=XW), in_axes=1)(bases)
      K = jax.vmap(
        lambda phi_i, dphi_i, phi_bdry_i: jax.vmap(
          lambda phi_j, dphi_j, phi_bdry_j: _bilinear_form(
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
      sol_coeff, _, _, _ = jnp.linalg.lstsq(K, F)
      return sol_coeff, K
    return _galerkin_solve

  def galerkin_lsq(self):
    _bilinear_form = self.pde.bilinear_form()
    _residual = self.pde.residual()
    def _galerkin_lsq(
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
        lambda v, dv, v_bdry: _residual(
          u=u, du=du, u_bdry=u_bdry, v=v, dv=dv, v_bdry=v_bdry, f=f, XW=XW, XW_bdry=XW_bdry
        ),
        in_axes=1
      )(net, dnet, net_bdry)

      K = jax.vmap(
        lambda v_i, dv_i, v_bdry_i: jax.vmap(
          lambda v_j, dv_j, v_bdry_j: _bilinear_form(
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
      coeff, _, _, _ = jnp.linalg.lstsq(K, F)
      return coeff
    return _galerkin_lsq


  def loss_fn(self, net_fn, dnet_fn) -> Callable:
    _galerkin_lsq = self.galerkin_lsq()
    _error_eta = self.pde.error_eta()
    _net_fn = net_fn
    _dnet_fn = dnet_fn
    def _loss_fn(
      params: optax.Params,
      domain_quad: Quadrature,
      u: FunctionState,
      f: jax.Array,
    ) -> tuple[float, jax.Array]:
      X_int_tensor = domain_quad.interior[:, jnp.newaxis]
      X_bdry_tensor = domain_quad.boundary[:, jnp.newaxis]
      net = FunctionState(
        interior=_net_fn(X_int_tensor, params),
        grad_interior=_dnet_fn(X_int_tensor, params),
        boundary=_net_fn(X_bdry_tensor, params)
      )
      v_nn_coeff = _galerkin_lsq(
        u=u.interior,
        du=u.grad_interior,
        u_bdry=u.boundary,
        net=net.interior,
        dnet=net.grad_interior,
        net_bdry=net.boundary,
        f=f,
        XW=domain_quad.interior_weights,
        XW_bdry=domain_quad.boundary_weights,
      )
      loss = _error_eta(
        u=u.interior,
        du=u.grad_interior,
        u_bdry=u.boundary,
        v=jnp.dot(net.interior, v_nn_coeff),
        dv=jnp.dot(net.grad_interior, v_nn_coeff),
        v_bdry=jnp.dot(net.boundary, v_nn_coeff),
        f=f,
        XW=domain_quad.interior_weights,
        XW_bdry=domain_quad.boundary_weights,
      )
      return -loss, v_nn_coeff  # It is maximizing!
    return _loss_fn




# def inner_product(u: jax.Array, v: jax.Array, XW: jax.Array) -> float:
# 	return jnp.sum(XW * u * v)


# def linear_op(f: jax.Array, v: jax.Array, XW: jax.Array) -> float:
# 	return inner_product(u=f, v=v, XW=XW)


# def bilinear_op(
# 	u: jax.Array,
# 	du: jax.Array,
# 	u_bdry: jax.Array,
# 	v: jax.Array,
# 	dv: jax.Array,
# 	v_bdry: jax.Array,
# 	XW: jax.Array,
# 	XW_bdry: jax.Array
# ) -> float:
# 	a1 = inner_product(u=du, v=dv, XW=XW)
# 	a2 = inner_product(u=u_bdry, v=v_bdry, XW=XW_bdry)
# 	eps = 1e-3
# 	return a1 + a2 / eps


# def norm(
# 	v: jax.Array,
# 	dv: jax.Array,
# 	v_bdry: jax.Array,
# 	XW: jax.Array,
# 	XW_bdry: jax.Array
# ) -> float:
# 	"""
# 	Norm |||v|||
# 	"""
# 	a = bilinear_op(
# 		u=v,
# 		v=v,
# 		du=dv,
# 		dv=dv,
# 		u_bdry=v_bdry,
# 		v_bdry=v_bdry,
# 		XW=XW,
# 		XW_bdry=XW_bdry
# 	)
# 	return jnp.sqrt(a)


# def residual(
# 	u: jax.Array,
# 	du: jax.Array,
# 	u_bdry: jax.Array,
# 	v: jax.Array,
# 	dv: jax.Array,
# 	v_bdry: jax.Array,
# 	f: jax.Array,
# 	XW: jax.Array,
# 	XW_bdry: jax.Array
# ) -> float:
# 	"""
# 	Residual L(v) - a(u, v)
# 	"""
# 	L_v = linear_op(f=f, v=v, XW=XW)
# 	a_uv = bilinear_op(
# 		u=u,
# 		v=v,
# 		du=du,
# 		dv=dv,
# 		u_bdry=u_bdry,
# 		v_bdry=v_bdry,
# 		XW=XW,
# 		XW_bdry=XW_bdry
# 	)
# 	return L_v - a_uv


# def error_eta(
# 	u: jax.Array,
# 	du: jax.Array,
# 	u_bdry: jax.Array,
# 	v: jax.Array,
# 	dv: jax.Array,
# 	v_bdry: jax.Array,
# 	f: jax.Array,
# 	XW: jax.Array,
# 	XW_bdry: jax.Array
# ) -> float:
# 	r"""Error approximation
# 	$$ \eta(u, v) = <r(u), v> / |||v||| = (L(v) - a(u, v)) / |||v||| $$
# 	"""
# 	norm_v = norm(
# 		v=v,
# 		dv=dv,
# 		v_bdry=v_bdry,
# 		XW=XW,
# 		XW_bdry=XW_bdry
# 	)
# 	res = residual(
# 		u=u,
# 		du=du,
# 		u_bdry=u_bdry,
# 		v=v,
# 		dv=dv,
# 		v_bdry=v_bdry,
# 		f=f,
# 		XW=XW,
# 		XW_bdry=XW_bdry
# 	)
# 	return res / norm_v



# # Galerkin Schemes
# def my_galerkin_solve(
# 	bases: Sequence[jax.Array],
# 	bases_bdry: Sequence[jax.Array],
# 	dbases: Sequence[jax.Array],
# 	f: jax.Array,
# 	XW: jax.Array,
# 	XW_bdry: jax.Array,
# ) -> jax.Array:

# 	F = jax.vmap(lambda v : linear_op(f=f, v=v, XW=XW), in_axes=1)(bases)
# 	K = jax.vmap(
# 		lambda phi_i, dphi_i, phi_bdry_i: jax.vmap(
# 			lambda phi_j, dphi_j, phi_bdry_j: bilinear_op(
# 				u=phi_i,
# 				du=dphi_i,
# 				u_bdry=phi_bdry_i,
# 				v=phi_j,
# 				dv=dphi_j,
# 				v_bdry=phi_bdry_j,
# 				XW=XW,
# 				XW_bdry=XW_bdry
# 			),
# 			in_axes=1
# 		)(bases, dbases, bases_bdry),
# 		in_axes=1
# 	)(bases, dbases, bases_bdry)
# 	sol_coeff, _, _, _ = jnp.linalg.lstsq(K, F) # Get solution coefficients
# 	return sol_coeff

