import jax
import jax.numpy as jnp

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from .domain import Geometry

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FunctionState:
  interior: jax.Array  # shape (num_interior, )
  grad_interior: jax.Array | None = None # shape (num_interior, dim)
  boundary: jax.Array | None = None# shape (num_boundary,)
  grad_boundary: jax.Array | None = None# shape (num_boundary, dim)


@dataclass(frozen=True)
class PDE(ABC):
  """Base class for all PDE."""
  domain: Geometry

  @abstractmethod
  def source(self) -> Callable:
    pass

  @abstractmethod
  def inner_product(self) -> Callable:
    pass

  @abstractmethod
  def bilinear_form(self) -> Callable:
    pass

  @abstractmethod
  def linear_operator(self) -> Callable:
    pass

  def weak_norm(self) -> Callable:
    _bilinear_form = self.bilinear_form()
    def _weak_norm(
      v: jax.Array,
      dv: jax.Array,
      v_bdry: jax.Array,
      XW: jax.Array,
      XW_bdry: jax.Array
    ) -> float:
      """
      Norm |||v|||
      """
      a = _bilinear_form(
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
    return _weak_norm

  def residual(self) -> Callable:
    _linear_operator = self.linear_operator()
    _bilinear_form = self.bilinear_form()
    def _residual(
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
      L_v = _linear_operator(f=f, v=v, XW=XW)
      a_uv = _bilinear_form(
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
    return _residual

  def error_eta(self) -> Callable:
    _weak_norm = self.weak_norm()
    _residual = self.residual()
    def _error_eta(
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
      norm_v = _weak_norm(
        v=v,
        dv=dv,
        v_bdry=v_bdry,
        XW=XW,
        XW_bdry=XW_bdry
      )
      res = _residual(
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
    return _error_eta