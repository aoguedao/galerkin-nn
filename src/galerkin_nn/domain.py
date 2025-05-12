"""
Geometry module for defining domains.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Quadrature:
  interior: jax.Array  # shape (num_interior, dim)
  interior_weights: jax.Array  # shape (num_interior,)
  boundary: jax.Array  # shape (num_boundary, dim)
  boundary_weights: jax.Array  # shape (num_boundary,)


@jax.tree_util.register_dataclass
@dataclass
class Geometry(ABC):
  """Base class for all geometries."""

  @abstractmethod
  def quadrature(self, name) -> Quadrature:
    """
    Returns Quadrature object, i.e., interior and boundary quadratures
    """

  @abstractmethod
  def _quadrature_interior(self) -> tuple[jax.Array, jax.Array]:
    """
    Returns interior quadrature nodes and weights
    """
    pass

  @abstractmethod
  def _quadrature_boundary(self) -> tuple[jax.Array, jax.Array]:
    """
    Returns boundary quadrature nodes and weights
    """
    pass


@jax.tree_util.register_dataclass
@dataclass
class IntervalGeom(Geometry):
  x_start: float
  x_end: float

  def quadrature(
    self,
    degree: int,
    name: str = "gauss-legendre"
  ) -> Quadrature:
    int_nodes, int_weights = self._quadrature_interior(name=name, degree=degree)
    bdry_nodes, bdry_weights = self._quadrature_boundary()
    quadrature = Quadrature(
      interior=int_nodes,
      interior_weights=int_weights,
      boundary=bdry_nodes,
      boundary_weights=bdry_weights
    )
    return quadrature

  def _quadrature_interior(
    self,
    degree: int,
    name: str = "gauss-legendre"
  ):
    if name.lower() == "gauss-legendre":
      nodes, weights = np.polynomial.legendre.leggauss(degree)
      midpoint = 0.5 * (self.x_start + self.x_end)
      halfwidth = 0.5 * (self.x_end - self.x_start)
      quad_nodes = jnp.array(midpoint + halfwidth * nodes)
      quad_weights = jnp.array(halfwidth * weights)
      return quad_nodes, quad_weights
    else:
      raise NotImplementedError(f"Interior quadrature {name} not yet implemented.")

  def _quadrature_boundary(self):
    bdry_nodes = jnp.array([self.x_start, self.x_end])
    bdry_weights = jnp.array([1.0, 1.0])
    return bdry_nodes, bdry_weights
