import jax
import jax.numpy as jnp
import optax

from dataclasses import dataclass
from functools import partial
from typing import Callable, Union, List, Sequence, Optional

from .formulations import FunctionState, PDE, DDPDE
from .quadratures import Quadrature
from .utils import make_u_fn, make_impedance_trace

TraceFn = Callable[[jax.Array], jax.Array]

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
  d = jnp.sqrt(jnp.clip(jnp.diag(K), 1e-30))          # (m,)
  Dinv = 1.0 / d

  F = jnp.reshape(F, (F.shape[0], -1))[:, :1]

  Khat = (K * Dinv[None, :]) * Dinv[:, None]          # D^{-1} K D^{-1}
  Fhat = F * Dinv[:, None]

  # 3) Ridge scaled to matrix size
  lam = ridge * (jnp.trace(Khat) / Khat.shape[0])
  Khat = Khat + lam * jnp.eye(Khat.shape[0], dtype=Khat.dtype)

  # 4) Cholesky solver
  L = jnp.linalg.cholesky(Khat)
  c_hat = jax.scipy.linalg.cho_solve((L, True), Fhat)

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
      F = residual(u=u, v=v, quad=quad)
      F = jnp.atleast_2d(F).T
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
        loss = error_eta(u=u, v=basis_net, quad=quad)  # shape (1, 1)
        loss = loss.squeeze()

        # jax.debug.print("E shape: {}", loss.shape)
        # loss = jnp.linalg.norm(loss)
        # jax.debug.print("loss shape: {}", loss.shape)  # should be ()
        
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
      grad_interior=jnp.einsum("nid,ij->njd", sigma_net.grad_interior, basis_coeff),
      grad_boundary=jnp.einsum("nid,ij->njd", sigma_net.grad_boundary, basis_coeff)
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
      F = linear_op(v=bases, quad=quad)
      F = jnp.atleast_2d(F).T
      K = bilinear_form(u=bases, v=bases, quad=quad)
      coeff, _, _, _ = jnp.linalg.lstsq(K, F)
      # coeff = spd_solve(K, F)
      # jax.debug.print("K shape {}", K.shape)
      # jax.debug.print("F shape before solve {}", F.shape)
      # jax.debug.print("u_coeff shape after solve {}", coeff.shape)
      # jax.debug.print("u.interior shape {}", (bases.interior @ (coeff if coeff.ndim==2 else coeff[:,None])).shape)
      assert F.ndim == 2 and K.ndim == 2, f"Bad shapes: F {F.shape}, K {K.shape}"
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
    error = jnp.inf
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

      # def lincomb_function_state(bases: FunctionState, coeff: jax.Array) -> FunctionState:
      #   c = coeff.reshape(-1)                               # (m,)
      #   # interior: (N, m) @ (m,) -> (N,)
      #   ui = jnp.einsum('nm,m->n', bases.interior, c)[:, None]         # (N,1)
      #   # boundary: (Nb, m) @ (m,) -> (Nb,)
      #   ub = jnp.einsum('am,m->a', bases.boundary, c)[:, None]         # (Nb,1)
      #   # grad: (N, m, d) • (m,) -> (N, d) -> expand to (N,1,d)
      #   gi = jnp.einsum('nmd,m->nd', bases.grad_interior, c)[:, None, :]  # (N,1,d)
      #   gb = jnp.einsum('nmd,m->nd', bases.grad_boundary, c)[:, None, :]
      #   return FunctionState(interior=ui, boundary=ub, grad_interior=gi, grad_boundary=gb)

      # u = lincomb_function_state(bases, u_coeff)  # single-state u
      print(f"\tEta error: {error}")
      bstep += 1

    return u, u_coeff, eta_errors, basis_list, basis_params_list, basis_coeff_list, sigma_net_fn_list


def _zeros_like_boundary(quad) -> TraceFn:
  return lambda X: jnp.zeros((X.shape[0], 1), dtype=X.dtype)


def _relax_fn(new_f: TraceFn, old_f: Optional[TraceFn], omega: float) -> TraceFn:
  if old_f is None or omega == 1.0:
    return new_f
  return lambda X: (1.0 - omega) * old_f(X) + omega * new_f(X)


class GalerkinNNDD:
  """
  Galerkin Neural Networks — Domain Decomposition (overlapping Schwarz).

  Parameters
  ----------
  base_pde : PDE | Sequence[PDE]
    Single PDE template (used for all subdomains) or a list per subdomain.
    For piecewise κ, pass a *template* and use `with_k` externally to make
    per-side PDEs before constructing this class.
  quadratures : Sequence[DDQuadrature]
    One quadrature per subdomain.
  eps_interface : float
    Robin/impedance parameter δ used on Γ_i via DDPDE.
  transmission : {'impedance','dirichlet'}
    'impedance' uses g = u_j + δ κ_i ∂_{n_i}u_j (recommended),
    'dirichlet' uses g = u_j.
  trace_relaxation : float
    Relaxation for traces (0 < omega ≤ 1).
  """

  def __init__(
    self,
    base_pde: Union[object, Sequence[object]],
    quadratures: Sequence[object],
    *,
    eps_interface: float = 1e-5,
    transmission: str = "impedance",
    trace_relaxation: float = 1.0
  ):
    self.Q: List[object] = list(quadratures)
    self.M: int = len(self.Q)
    if self.M < 2:
      raise ValueError("Need at least two subdomains.")

    if isinstance(base_pde, (list, tuple)):
      if len(base_pde) != self.M:
        raise ValueError("len(base_pde) must match len(quadratures).")
      self.PDE = list(base_pde)
    else:
      # broadcast the same PDE to all subdomains (OK for homogeneous κ)
      self.PDE = [base_pde] * self.M

    self.eps_interface = float(eps_interface)
    self.transmission = transmission.lower()
    if self.transmission not in ("impedance", "dirichlet"):
      raise ValueError("transmission must be 'impedance' or 'dirichlet'")
    self.trace_relaxation = float(trace_relaxation)

    # traces per target i: tuple of length = Q[i].neighbor_ids
    self._g_per_target: List[List[Optional[TraceFn]]] = [
      [None] * getattr(Qi, "boundary_owner_onehot").shape[1] for Qi in self.Q
    ]

  def solve(
    self,
    *,
    net_fn,
    activations_fn,
    network_widths_fn,
    learning_rates_fn,
    max_bases: int,
    max_epoch_basis: int,
    tol_solution: float,
    tol_basis: float,
    seeds: Optional[Sequence[int]] = None,
    max_sweeps: int = 8,
    tol_jump: float = 5e-4,
    init_states: Optional[Sequence[object]] = None
  ):
    """
    Runs Gauss-Seidel Schwarz sweeps across all subdomains.
    Returns:
      dict(u_fns=[...], logs=[...])
    """
    M = self.M
    Q = self.Q
    PDE = self.PDE
    eps_int = self.eps_interface

    # seeds per subdomain
    if seeds is None:
      seeds = [42 + 100*i for i in range(M)]
    else:
      if len(seeds) != M:
        raise ValueError("len(seeds) must match number of subdomains.")

    # initial states: zero FunctionState on each subdomain
    if init_states is None:
      z = lambda X: jnp.zeros((X.shape[0], 1), dtype=X.dtype)
      gradz = lambda X: jnp.zeros((X.shape[0], 1, Q[0].dim), dtype=X.dtype)
      states = [FunctionState.from_function(z, Qi, gradz) for Qi in Q]
    else:
      states = list(init_states)

    # initialize trace lists with zero functions where needed
    for i in range(M):
      onehot = getattr(Q[i], "boundary_owner_onehot")
      Jn = onehot.shape[1]
      for jcol in range(Jn):
        if self._g_per_target[i][jcol] is None:
          self._g_per_target[i][jcol] = _zeros_like_boundary(Q[i])

    # place-holders for current u_fn per subdomain
    u_fns: List[Callable[[jax.Array], jax.Array]] = [
      _zeros_like_boundary(Q[i]) for i in range(M)
    ]

    history = []
    # main Schwarz sweeps (Gauss–Seidel)
    for k in range(max_sweeps):
      for i in range(M):
        # assemble DDPDE on target i with current trace tuple (ordered by neighbor_ids)
        trace_tuple = tuple(self._g_per_target[i])
        pde_i = DDPDE(base=PDE[i], eps_interface=eps_int, trace_fns=trace_tuple)

        solver_i = GalerkinNN(pde_i, Q[i])
        out = solver_i.solve(
          seed=seeds[i] + 100*k,
          u0=states[i],
          net_fn=net_fn,
          activations_fn=activations_fn,
          network_widths_fn=network_widths_fn,
          learning_rates_fn=learning_rates_fn,
          max_bases=max_bases,
          max_epoch_basis=max_epoch_basis,
          tol_solution=tol_solution,
          tol_basis=tol_basis,
        )
        u_state_out_i, u_coeff_i, *_rest, basis_coeff_list_i, sigma_net_fn_list_i = out
        u_fn_i = make_u_fn(sigma_net_fn_list_i, u_coeff=u_coeff_i, basis_coeff_list=basis_coeff_list_i)
        u_fns[i] = u_fn_i  # store latest

        # update neighbor traces that depend on subdomain i
        self._update_neighbor_traces_from(i, u_fn_i)

      # optional diagnostic for 2-subdomain 1D case (interface jump)
      if M == 2 and Q[0].dim == 1:
        x_if_R = Q[0].boundary_x[-1:]
        x_if_L = Q[1].boundary_x[:1]
        u0_if = jnp.squeeze(u_fns[0](x_if_R)).item()
        u1_if = jnp.squeeze(u_fns[1](x_if_L)).item()
        jump  = u0_if - u1_if
        history.append(dict(sweep=k+1, u0_if=u0_if, u1_if=u1_if, jump=jump))
        print(f"[sweep {k+1:02d}] u0(b0^+)={u0_if:+.6e}, u1(a1^-)={u1_if:+.6e}, jump={jump:+.6e}")
        if abs(jump) < tol_jump:
          break

    return dict(u_fns=u_fns, logs=history)


  def _update_neighbor_traces_from(self, src_idx: int, u_fn_src: Callable[[jax.Array], jax.Array]):
    """
    After solving on Ω_src, build traces for every *target* Ω_tgt that lists src_idx
    among its neighbor_ids. Impedance uses κ from the *target* PDE.
    """
    for tgt_idx, Qt in enumerate(self.Q):
      # find which column corresponds to src_idx in target's onehot
      nids = getattr(Qt, "neighbor_ids")
      if nids is None:
        # assume pairwise 2-subdomain setup; if not, skip
        continue
      if src_idx not in nids:
        continue
      jcol = nids.index(src_idx)  # column in boundary_owner_onehot / trace_fns

      # build new trace for target
      if self.transmission == "impedance":
        # target-side κ via k(X) of PDE[tgt_idx]
        kappa_t = self.PDE[tgt_idx].k
        g_new = make_impedance_trace(u_fn_src, Qt, kappa_t, self.eps_interface)
      else:
        g_new = lambda X, f=u_fn_src: f(X)

      # relax into current stored trace
      g_old = self._g_per_target[tgt_idx][jcol]
      self._g_per_target[tgt_idx][jcol] = _relax_fn(g_new, g_old, self.trace_relaxation)  