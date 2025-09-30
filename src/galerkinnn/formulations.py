import jax
import jax.numpy as jnp

from abc import ABC, abstractmethod
from flax import struct
# from dataclasses import dataclass
from typing import Callable, Tuple

from .quadratures import Quadrature

TraceFn = Callable[[jax.Array], jax.Array]   # X -> (N,1) or (N,)


@struct.dataclass
class FunctionState:
  interior: jax.Array       # (N, n_states)
  boundary: jax.Array       # (Nb, n_states)
  grad_interior: jax.Array  # (N, n_states, dim)
  grad_boundary: jax.Array  # (Nb, n_states, dim)

  @staticmethod
  def _ensure_2d_values(vals: jax.Array) -> jax.Array:
    return vals if vals.ndim == 2 else vals[:, None]

  @staticmethod
  def _ensure_grad_3d(grads: jax.Array) -> jax.Array:
    return grads if grads.ndim == 3 else grads[:, None, :]

  @classmethod
  def from_function(
    cls,
    func: Callable[[jax.Array], jax.Array],
    quad: Quadrature,
    grad_func: Callable[[jax.Array], jax.Array] | None = None,
    use_forward_jac: bool = True,
  ) -> "FunctionState":
    u_int = func(quad.interior_x)  # (N, n_states)
    u_bnd = func(quad.boundary_x)  # (Nb, n_states)

    if grad_func is None:
      grad_single = jax.jacfwd(func) if use_forward_jac else jax.jacrev(func)
      grad_func = jax.vmap(grad_single)

    g_int = grad_func(quad.interior_x)  # (N, n_states, dim)
    g_bnd = grad_func(quad.boundary_x)  # (Nb, n_states, dim)

    state = cls(
      interior=cls._ensure_2d_values(u_int),
      boundary=cls._ensure_2d_values(u_bnd),
      grad_interior=cls._ensure_grad_3d(g_int),
      grad_boundary=cls._ensure_grad_3d(g_bnd)
    )
    return state

  @property
  def n_states(self) -> int:
    return self.interior.shape[1]


@struct.dataclass
class PDE(ABC):

  @abstractmethod
  def source(self) -> Callable[[jax.Array], jax.Array]: ...

  @abstractmethod
  def bilinear_form(self) -> Callable: ...

  @abstractmethod
  def linear_operator(self) -> Callable: ...

  @abstractmethod
  def energy_norm(self) -> Callable: ...

  def residual(self) -> Callable:
    _L  = self.linear_operator()
    _a  = self.bilinear_form()

    def _residual(u: FunctionState, v: FunctionState, quad) -> jax.Array:
      """
      Residual matrix R(u,v) with shape (n_u, n_v):
        R_ij = L(v_j) - a(u_i, v_j)
      """
      Lv  = _L(v=v, quad=quad).reshape(-1)            # (n_v,) <- squeeze row to 1-D
      AuV = _a(u=u, v=v, quad=quad)                   # (n_u, n_v)
      return Lv[None, :] - AuV                        # (n_u, n_v)
    return _residual

  def error_eta(self) -> Callable:
    _norm = self.energy_norm()
    _res  = self.residual()

    def _error_eta(u: FunctionState, v: FunctionState, quad) -> jax.Array:
      """
      Elementwise error indicator matrix E(u,v) with shape (n_u, n_v):
        E_ij = (L(v_j) - a(u_i, v_j)) / |||v_j|||
      """
      R  = _res(u=u, v=v, quad=quad)                  # (n_u, n_v)
      nv = _norm(v=v, quad=quad)                      # (n_v,)
      denom = jnp.maximum(nv, jnp.array(1e-12, nv.dtype))
      return R / denom[None, :]                       # (n_u, n_v)
    return _error_eta


@struct.dataclass
class DDPDE(PDE):
  """
  ASM penalty transmission wrapper (symmetric, PSD).
  - base: underlying PDE (keeps physical Robin on Γ_phys)
  - eps_interface: interface penalty (Dirichlet ≈ 1/eps_interface → ∞)
  - trace_fns: STATIC tuple of neighbor trace callables ordered as quad.neighbor_ids;
               each returns (N,1) or (N,) when evaluated at quad.boundary_x
  """
  base: PDE
  eps_interface: float = 1e-3
  trace_fns: Tuple[TraceFn, ...] = struct.field(pytree_node=False, default_factory=tuple)

  # (optional) convenience to update traces without changing dataclass layout
  def with_traces(self, trace_fns: Tuple[TraceFn, ...]) -> "DDPDE":
    return self.replace(trace_fns=trace_fns)

  # passthrough
  def source(self):
    return self.base.source()

  @staticmethod
  def _interface_mask(quad: Quadrature) -> jax.Array:
    """(Nb,) bool: True on artificial (non-global) boundary; zeros if not DD."""
    mask_g = getattr(quad, "boundary_mask_global", None)
    if mask_g is None:
      return jnp.zeros((quad.boundary_x.shape[0],), dtype=bool)
    return ~mask_g

  def _g_on_boundary(self, quad: Quadrature) -> jax.Array:
    """Assemble neighbor trace g at THIS subdomain boundary via one-hot ownership."""
    Nb = quad.boundary_x.shape[0]
    onehot = getattr(quad, "boundary_owner_onehot", None)
    neighbor_ids = getattr(quad, "neighbor_ids", ())
    if onehot is None or len(neighbor_ids) == 0 or len(self.trace_fns) == 0:
      return jnp.zeros((Nb,), dtype=quad.boundary_w.dtype)

    Jn = onehot.shape[1]
    assert len(self.trace_fns) == Jn, "trace_fns length must match neighbor_ids length"

    cols = [fn(quad.boundary_x).reshape(-1) for fn in self.trace_fns]   # (Nb,)
    G = jnp.stack(cols, axis=1)                                         # (Nb, Jn)
    g_pick = jnp.sum(G * onehot, axis=1)                                # (Nb,)
    return g_pick * self._interface_mask(quad).astype(g_pick.dtype)     # zero on global rows

  def bilinear_form(self):
    a_base = self.base.bilinear_form()
    inv_eps_int = 1.0 / self.eps_interface

    def a(u: FunctionState, v: FunctionState, quad: Quadrature) -> jax.Array:
      A0 = a_base(u=u, v=v, quad=quad)
      # interface penalty (branch-free)
      mask_int = self._interface_mask(quad).astype(quad.boundary_w.dtype)    # (Nb,)
      gamma_int = inv_eps_int * quad.boundary_w * mask_int                   # (Nb,)
      Aint = jnp.einsum("an,am,a->nm", u.boundary, v.boundary, gamma_int)    # (n_u, n_v)
      return A0 + Aint
    return a

  def linear_operator(self):
    L_base = self.base.linear_operator()
    inv_eps_int = 1.0 / self.eps_interface

    def L(v: FunctionState, quad: Quadrature) -> jax.Array:
      L0 = L_base(v=v, quad=quad)
      mask_int = self._interface_mask(quad).astype(quad.boundary_w.dtype)    # (Nb,)
      g_b = self._g_on_boundary(quad)                                        # (Nb,)
      h_int = inv_eps_int * g_b                                              # (Nb,)
      Fint = jnp.einsum("a,an,a->n", h_int, v.boundary, quad.boundary_w * mask_int)
      return L0 + Fint
    return L

  def energy_norm(self):
    norm_base = self.base.energy_norm()
    inv_eps_int = 1.0 / self.eps_interface

    def norm(v: FunctionState, quad: Quadrature) -> jax.Array:
      nb = norm_base(v=v, quad=quad)             # sqrt of base energy
      nb2 = nb * nb
      mask_int = self._interface_mask(quad).astype(quad.boundary_w.dtype)
      gamma_int = inv_eps_int * quad.boundary_w * mask_int
      add = jnp.einsum("a,an->n", gamma_int, v.boundary**2)
      return jnp.sqrt(jnp.maximum(nb2 + add, 0.0))
    return norm
