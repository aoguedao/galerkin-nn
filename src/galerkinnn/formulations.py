import jax
import jax.numpy as jnp

from abc import ABC, abstractmethod
from flax import struct
from typing import Callable, Tuple

from .quadratures import Quadrature
from .utils import _dummy_quadrature, _dummy_state

TraceFn = Callable[[jax.Array], jax.Array]

_PDE_SHAPE_TEST_DIMS = (1, 2)


@struct.dataclass
class FunctionState:
  interior: jax.Array       # (N, n_states)
  boundary: jax.Array       # (Nb, n_states)
  grad_interior: jax.Array  # (N, n_states, dim)
  grad_boundary: jax.Array  # (Nb, n_states, dim)

  def __post_init__(self):
    """
    Lightweight shape check: do not modify fields, only validate consistency.
    """
    interior = self.interior
    boundary = self.boundary
    grad_interior = self.grad_interior
    grad_boundary = self.grad_boundary

    if interior.ndim != 2:
      raise ValueError(f"interior must be 2D (N, n_states), got shape {interior.shape}")
    if boundary.ndim != 2:
      raise ValueError(f"boundary must be 2D (Nb, n_states), got shape {boundary.shape}")
    if grad_interior.ndim != 3:
      raise ValueError(f"grad_interior must be 3D (N, n_states, dim), got shape {grad_interior.shape}")
    if grad_boundary.ndim != 3:
      raise ValueError(f"grad_boundary must be 3D (Nb, n_states, dim), got shape {grad_boundary.shape}")

    Ni, n_states = interior.shape
    Nb = boundary.shape[0]
    dim = grad_interior.shape[2]

    if boundary.shape[1] != n_states:
      raise ValueError(f"boundary n_states={boundary.shape[1]} != interior n_states={n_states}")
    if grad_interior.shape != (Ni, n_states, dim):
      raise ValueError(f"grad_interior shape {grad_interior.shape} incompatible with interior ({Ni}, {n_states}) and dim {dim}")
    if grad_boundary.shape != (Nb, n_states, dim):
      raise ValueError(f"grad_boundary shape {grad_boundary.shape} incompatible with boundary ({Nb}, {n_states}) and dim {dim}")

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

  def _shape_self_test(self):
    """
    Run a small shape sanity check on instantiation (opt-in via env var).
    Tries a couple of (n_u, n_v) combinations and candidate dimensions.
    """
    def _assert_shape(name: str, arr: jax.Array, expected: Tuple[Tuple[int, ...], ...]):
      shape = tuple(arr.shape)
      if shape not in expected:
        raise ValueError(f"{name} returned shape {shape}, expected one of {expected}")

    combos = ((1, 1), (1, 2), (2, 1))
    last_error = None
    for dim in _PDE_SHAPE_TEST_DIMS:
      quad = _dummy_quadrature(dim)
      try:
        for n_u, n_v in combos:
          u = _dummy_state(quad, n_states=n_u)
          v = _dummy_state(quad, n_states=n_v)
          L = self.linear_operator()(v=v, quad=quad)
          A = self.bilinear_form()(u=u, v=v, quad=quad)
          n = self.energy_norm()(v=v, quad=quad)
          R = self.residual()(u=u, v=v, quad=quad)
          E = self.error_eta()(u=u, v=v, quad=quad)

          _assert_shape("linear_operator", L, ((n_v,), (1, n_v)))
          _assert_shape("bilinear_form", A, ((n_u, n_v),))
          _assert_shape("energy_norm", n, ((n_v,), (n_v, 1)))
          _assert_shape("residual", R, ((n_u, n_v),))
          _assert_shape("error_eta", E, ((n_u, n_v),))
        return
      except Exception as exc:
        last_error = exc
        continue

    raise RuntimeError(f"PDE shape self-test failed: {last_error}") from last_error

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
  - trace_fns: STATIC tuple of neighbor trace callables ordered as quad.neighbor_ids; each returns (N,1) or (N,) when evaluated at quad.boundary_x
  """
  base: PDE
  eps_interface: float = 1e-3
  trace_fns: Tuple[TraceFn, ...] = struct.field(pytree_node=False, default_factory=tuple)

  # convenience to update traces without changing dataclass layout
  def with_traces(self, trace_fns: Tuple[TraceFn, ...]) -> "DDPDE":
    return self.replace(trace_fns=trace_fns)

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
    """
    Select neighbor-trace data on this boundary using the ownership one-hot.
    No masking here; masking happens in the forms.
    """
    Nb = quad.boundary_x.shape[0]
    onehot = getattr(quad, "boundary_owner_onehot", None)
    neighbor_ids = getattr(quad, "neighbor_ids", None)
    if onehot is None or onehot.shape[0] != Nb or not self.trace_fns or neighbor_ids is None:
      return jnp.zeros((Nb,), dtype=quad.boundary_w.dtype)

    if len(self.trace_fns) != len(neighbor_ids):
      raise ValueError("DDPDE.trace_fns must align with quad.neighbor_ids.")

    if onehot.shape[1] != len(neighbor_ids):
      raise ValueError("boundary_owner_onehot column count must match neighbor_ids.")

    cols = [fn(quad.boundary_x).reshape(-1) for fn in self.trace_fns]
    G = jnp.stack(cols, axis=1)
    return jnp.sum(G * onehot, axis=1)

  def _interface_mask(self, quad):
    # True on interface Γ, False on physical global boundary
    mask_g = getattr(quad, "boundary_mask_global", None)
    if mask_g is None:
      # If your DDQuadrature always sets this, you can raise instead.
      # Fallback: assume entire boundary is interface (safe for tests).
      Nb = quad.boundary_x.shape[0]
      return jnp.ones((Nb,), dtype=bool)
    return ~mask_g

  def bilinear_form(self):
    a_base = self.base.bilinear_form() if hasattr(self.base, "bilinear_form") else self.base.bilinear()
    inv_eps = 1.0 / self.eps_interface

    def a(u, v, quad):
      A0 = a_base(u=u, v=v, quad=quad)
      # interface mask: True on artificial boundary (NOT global physical)
      mask_g = getattr(quad, "boundary_mask_global", None)
      mask_int = (1 - mask_g.astype(quad.boundary_w.dtype)) if mask_g is not None else jnp.ones_like(quad.boundary_w)

      gamma_int = (1.0 / self.eps_interface) * quad.boundary_w * mask_int
      Aint = jnp.einsum("an,am,a->nm", u.boundary, v.boundary, gamma_int)
      return A0 + Aint
    return a

  def linear_operator(self):
    L_base = self.base.linear_operator()
    inv_eps = 1.0 / self.eps_interface

    def L(v, quad):
      L0  = L_base(v=v, quad=quad)
      g_b = self._g_on_boundary(quad)
      mask_g = getattr(quad, "boundary_mask_global", None)
      mask_int = (1 - mask_g.astype(quad.boundary_w.dtype)) if mask_g is not None else jnp.ones_like(quad.boundary_w)

      L_int = jnp.einsum("a,an,a->n", (1.0 / self.eps_interface) * g_b, v.boundary, quad.boundary_w * mask_int)
      return L0 + L_int
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
