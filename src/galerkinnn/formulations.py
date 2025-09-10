import jax
import jax.numpy as jnp

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from .quadratures import Quadrature


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
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


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
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
