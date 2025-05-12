import pytest
from galerkin_nn.domain import IntervalGeom

def test_quadrature_interior():
  interval = IntervalGeom(
    x_start=0.0,
    x_end=1.0,
    quadrature_name="gauss-legendre"
  )
  nodes, weights = interval.quadrature_interior(degree=10)
  assert nodes.shape == (10, )
  assert weights.shape == (10, )