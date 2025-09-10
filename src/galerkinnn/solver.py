import jax
import jax.numpy as jnp
import optax

from dataclasses import dataclass
from functools import partial
from typing import Callable

from .formulations import PDE, FunctionState
from .quadratures import Quadrature

jax.config.update("jax_enable_x64", True)


def spd_solve(
  K: jax.Array,
  F: jax.Array,
  ridge: float = 1e-8
) -> jax.Array:
  """
  Solve K c = F with K nominally SPD but possibly ill-conditioned.
  Returns c with same dtype as K/F. Shapes: K (m,m), F (m,1).
  """
  # 1) Symmetrize (remove tiny asymmetry from numerics)
  K = 0.5 * (K + K.T)

  # 2) Diagonal scaling (Jacobi preconditioning)
  d = jnp.sqrt(jnp.clip(jnp.diag(K), 1e-30))             # (m,)
  Dinv = 1.0 / d
  Khat = (K * Dinv[None, :]) * Dinv[:, None]             # D^{-1} K D^{-1}
  Fhat = F * Dinv[:, None]

  # 3) Ridge scaled to matrix size
  lam = ridge * (jnp.trace(Khat) / Khat.shape[0])
  Khat = Khat + lam * jnp.eye(Khat.shape[0], dtype=Khat.dtype)

  # 4) Cholesky solve
  L = jnp.linalg.cholesky(Khat)
  y = jax.scipy.linalg.solve_triangular(L, Fhat, lower=True)
  c_hat = jax.scipy.linalg.solve_triangular(L.T, y, lower=False)

  # 5) Undo scaling: c = D^{-1} c_hat
  return c_hat * Dinv[:, None]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class GalerkinNN:
  pde: PDE
  quad: Quadrature

  @staticmethod
  def augment_basis(
    u: FunctionState,
    quad: Quadrature,
    pde: PDE,
    sigma_net_fn: Callable[[jax.Array, optax.Params], jax.Array],
    params_init: optax.Params,
    learning_rate: float,
    max_epoch_basis: int,
    tol_basis: float,
  ):

    def galerkin_lsq(u: FunctionState, v: FunctionState, quad: Quadrature, pde: PDE) -> jax.Array:
      bilinear_form = pde.bilinear_form()
      residual = pde.residual()
      F = residual(u=u, v=v, quad=quad).T
      K = bilinear_form(u=v, v=v, quad=quad)
      coeff, _, _, _ = jnp.linalg.lstsq(K, F)
      # coeff = spd_solve(K, F)
      return coeff

    error_eta = pde.error_eta()
    sigma_net_fn_ckpt = lambda X, params: jax.checkpoint(sigma_net_fn)(X=X, params=params)  # new, testing
    sigma_net_grad = jax.vmap(jax.jacfwd(sigma_net_fn, argnums=0), in_axes=(0, None))

    # Optimization initialization
    params = params_init
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    @partial(jax.jit, static_argnames=['optimizer'], donate_argnums=(1, 2))
    def train_step(
      optimizer: optax.GradientTransformation,
      opt_state: optax.OptState,
      params: optax.Params,
      u: FunctionState,
      quad: Quadrature,
      pde: PDE
    ):

      def loss_fn(params):
        X_int = quad.interior_x
        X_bdry = quad.boundary_x
        sigma_net = FunctionState(
          interior=sigma_net_fn_ckpt(X=X_int, params=params),
          boundary=sigma_net_fn_ckpt(X=X_bdry, params=params),
          grad_interior=sigma_net_grad(X_int, params),
          grad_boundary=sigma_net_grad(X_bdry, params),
        )
        basis_coeff = galerkin_lsq(u, sigma_net, quad, pde)
        basis_coeff = jax.lax.stop_gradient(basis_coeff)
        basis_net = FunctionState(
          interior=sigma_net.interior @ basis_coeff,
          boundary=sigma_net.boundary @ basis_coeff,
          grad_interior=jnp.einsum("nid,ij->njd", sigma_net.grad_interior, basis_coeff),
          grad_boundary=jnp.einsum("nid,ij->njd", sigma_net.grad_boundary, basis_coeff)
        )
        loss = error_eta(u=u, v=basis_net, quad=quad).squeeze()
        return -jnp.abs(loss)

      loss, loss_grads = jax.value_and_grad(loss_fn, argnums=0)(params)
      updates, opt_state = optimizer.update(loss_grads, opt_state, params)
      params = optax.apply_updates(params, updates)
      return opt_state, params, loss, loss_grads

    loss_prev = jnp.inf
    for i in range(max_epoch_basis + 1):
      opt_state, params, loss, loss_grads = train_step(
        optimizer=optimizer,
        opt_state=opt_state,
        params=params,
        u=u,
        quad=quad,
        pde=pde
      )
      if i % (max_epoch_basis // 10) == 0:
        print(f"step {i}, loss: {- float(loss):.4e}, grad norm: {optax.global_norm(loss_grads):.4e}")
      if jnp.abs((loss - loss_prev) / loss_prev) < tol_basis:
        break
      else:
        loss_prev = loss

    # Final basis
    sigma_net = FunctionState(
      interior=sigma_net_fn_ckpt(X=quad.interior_x, params=params),
      boundary=sigma_net_fn_ckpt(X=quad.boundary_x, params=params),
      grad_interior=sigma_net_grad(quad.interior_x, params),
      grad_boundary=sigma_net_grad(quad.boundary_x, params)
    )
    basis_coeff = galerkin_lsq(u, sigma_net, quad, pde)
    basis_net = FunctionState(
      interior=sigma_net.interior @ basis_coeff,
      boundary=sigma_net.boundary @ basis_coeff,
      grad_interior=jnp.einsum('nid,ij->njd', sigma_net.grad_interior, basis_coeff),
      grad_boundary=jnp.einsum('nid,ij->njd', sigma_net.grad_boundary, basis_coeff)
    )
    eta = error_eta(u=u, v=basis_net, quad=quad).squeeze()
    fixed_basis_net_fn = partial(sigma_net_fn, params=params)
    return eta, basis_net, params, basis_coeff, fixed_basis_net_fn

  def solve(
    self,
    seed: int,
    u0: FunctionState,
    net_fn: Callable,
    activations_fn: Callable[[int], Callable[[jax.Array], jax.Array]],
    network_widths_fn: Callable[[int], int],
    learning_rates_fn: Callable[[int], float],
    max_bases: int = 6,
    max_epoch_basis: int = 100,
    tol_solution: float = 1e-8,
    tol_basis: float = 1e-6,
  ):
    def galerkin_solve(bases: FunctionState, quad: Quadrature, pde: PDE) -> jax.Array:
      linear_op = pde.linear_operator()
      bilinear_form = pde.bilinear_form()
      F = linear_op(v=bases, quad=quad).T
      K = bilinear_form(u=bases, v=bases, quad=quad)
      # coeff, _, _, _ = jnp.linalg.lstsq(K, F)
      coeff = spd_solve(K, F)
      return coeff

    eta_errors = []
    basis_list = []
    basis_params_list = []
    basis_coeff_list = []
    sigma_net_fn_list = []

    key = jax.random.key(seed)
    pde = self.pde
    quad = self.quad

    u = u0
    bstep = 1
    error = 1e10
    while (error > tol_solution) and (bstep <= max_bases):
      print(f"Basis Step: {bstep}")
      activation = activations_fn(bstep)
      neurons = network_widths_fn(bstep)
      learning_rate = learning_rates_fn(bstep)
      sigma_net_fn = partial(net_fn, activation=activation)

      key_W, key_b, key = jax.random.split(key, num=3)
      initializer_W = jax.nn.initializers.glorot_normal()
      params_init = {
        # "W": jnp.ones(shape=(quad.dim, neurons)),
        # "W": initializer_W(key=key_W, shape=(quad.dim, neurons)),
        "W": jax.random.normal(shape=(quad.dim, neurons), key=key_W),
        "b": - jnp.linspace(0, 1, neurons),
      }
      eta, basis, basis_params, basis_coeff, sigma_net_fn =  self.augment_basis(
        u=u,
        quad=quad,
        pde=pde,
        sigma_net_fn=sigma_net_fn,
        params_init=params_init,
        learning_rate=learning_rate,
        max_epoch_basis=max_epoch_basis,
        tol_basis=tol_basis
      )

      error = jnp.abs(eta)
      eta_errors.append(error)
      basis_list.append(basis)
      basis_params_list.append(basis_params)
      basis_coeff_list.append(basis_coeff)
      sigma_net_fn_list.append(sigma_net_fn)

      bases = FunctionState(
        interior=jnp.concatenate([phi.interior for phi in basis_list], axis=1),
        boundary=jnp.concatenate([phi.boundary for phi in basis_list], axis=1),
        grad_interior=jnp.concatenate([phi.grad_interior for phi in basis_list], axis=1),
        grad_boundary=jnp.concatenate([phi.grad_boundary for phi in basis_list], axis=1)
      )

      u_coeff = galerkin_solve(bases, quad, pde)
      u = FunctionState(
        interior=bases.interior @ u_coeff,
        boundary=bases.boundary @ u_coeff,
        grad_interior=jnp.einsum("xbd,bj->xjd", bases.grad_interior, u_coeff),
        grad_boundary=jnp.einsum("xbd,bj->xjd", bases.grad_boundary, u_coeff)
      )

      print(f"\tEta error: {error}")
      bstep += 1

    return u, u_coeff, eta_errors, basis_list, basis_params_list, basis_coeff_list, sigma_net_fn_list
