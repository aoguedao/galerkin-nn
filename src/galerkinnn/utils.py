import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from typing import Callable, Optional, Tuple, Literal, Sequence

from .quadratures import Quadrature

UFn   = Callable[[jax.Array], jax.Array]  # X -> (N,1)
SigmaFn = Callable[..., jax.Array]  # expects fn(X=...) -> (N, m_i) or (N,1)

# ---------------------------
# Reconstruct u(X) callable
# ---------------------------

def make_u_fn(
  sigma_net_fn_list: Sequence[Callable[[jax.Array], jax.Array]],
  u_coeff: jax.Array,                          # (B,) or (B,1)
  basis_coeff_list: Sequence[jax.Array],       # each (m_i,) or (m_i,1)
) -> Callable[[jax.Array], jax.Array]:
  """
  Build u(X) = sum_i [ (σ_i(X) @ α_i) * c_i ].

  σ_i : ℝ^{N×m_i},  α_i : ℝ^{m_i×1},  c_i : scalar
  Returns u_fn(X): ℝ^{N×1}.
  """
  B = len(sigma_net_fn_list)

  u_coeff = jnp.asarray(u_coeff)
  if u_coeff.ndim == 2:
    assert u_coeff.shape[1] == 1
    u_coeff = u_coeff[:, 0]
  assert u_coeff.shape[0] == B, f"u_coeff has {u_coeff.shape[0]} entries, expected {B}"

  alpha_list = [jnp.asarray(ai).reshape(-1, 1) for ai in basis_coeff_list]

  @jax.jit
  def u_fn(X: jax.Array) -> jax.Array:
    y = jnp.zeros((X.shape[0], 1), dtype=X.dtype)
    for fn, ai, ci in zip(sigma_net_fn_list, alpha_list, u_coeff):
      phi_i = fn(X=X) @ ai  # (N,1)
      y += phi_i * ci
    return y

  return u_fn


def make_u_and_grad_fn(
  sigma_net_fn_list: Sequence[Callable[[jax.Array], jax.Array]],
  u_coeff: jax.Array,
  basis_coeff_list: Sequence[jax.Array],
) -> Tuple[Callable[[jax.Array], jax.Array], Callable[[jax.Array], jax.Array]]:
  """
  Returns (u_fn, grad_u_fn) where both are JAX-differentiable.

  u(X) = sum_i [ (σ_i(X) @ α_i) * c_i ].
  grad_u(X) = ∂ₓu(X) with shape (N,1,dim).
  """
  B = len(sigma_net_fn_list)

  u_coeff = jnp.asarray(u_coeff)
  if u_coeff.ndim == 2:
    assert u_coeff.shape[1] == 1
    u_coeff = u_coeff[:, 0]
  assert u_coeff.shape[0] == B, f"u_coeff has {u_coeff.shape[0]} entries, expected {B}"

  alpha_list = [jnp.asarray(ai).reshape(-1, 1) for ai in basis_coeff_list]

  def u_fn(X: jax.Array) -> jax.Array:
    y = jnp.zeros((X.shape[0], 1), dtype=X.dtype)
    for fn, ai, ci in zip(sigma_net_fn_list, alpha_list, u_coeff):
      phi_i = fn(X=X) @ ai
      y += phi_i * ci
    return y

  def grad_u_fn(X: jax.Array) -> jax.Array:
    grads = []
    for fn, ai, ci in zip(sigma_net_fn_list, alpha_list, u_coeff):
      # JAX requires positional arg for jacfwd, so wrap in lambda with positional X
      G = jax.jacfwd(lambda X_: fn(X=X_))(X)  # (N,m_i,dim)
      g_i = jnp.einsum("nmd,mi->nid", G, ai) * ci  # (N,1,dim)
      grads.append(g_i)
    return jnp.sum(jnp.stack(grads, axis=0), axis=0)  # (N,1,dim)

  return jax.jit(u_fn), jax.jit(grad_u_fn)

def make_impedance_trace(u_fn, quad_target, kappa_fn, delta):
  a_t = float(quad_target.boundary_x[0, 0])
  b_t = float(quad_target.boundary_x[-1, 0])

  def n_from_X(X):
    x = X.reshape(-1)
    is_left  = jnp.isclose(x, a_t, atol=1e-12)
    is_right = jnp.isclose(x, b_t, atol=1e-12)
    # outward: left = -1, right = +1
    return (is_right.astype(X.dtype) - is_left.astype(X.dtype)).reshape(-1, 1)

  def u_scalar(x):
    return u_fn(x[None, :]).reshape(())
  grad_batch = jax.vmap(jax.grad(u_scalar))

  @jax.jit
  def g(X):
    uval = u_fn(X).reshape(-1, 1)                    # (Nb,1)
    du   = grad_batch(X).reshape(X.shape[0], 1)       # (Nb,1) in 1D
    n    = n_from_X(X)                                # (Nb,1)
    kn   = kappa_fn(X).reshape(-1, 1)                 # (Nb,1)  -- TARGET κ
    return uval + delta * kn * (du * n)               # (Nb,1)
  return g


# ---------------------------
# Shape-test helpers (private)
# ---------------------------
def _dummy_quadrature(dim: int, n_interior: int = 3, n_boundary: int = 2) -> Quadrature:
  """
  Lightweight quadrature stub for shape checking; values are placeholders.
  """
  interior_x = jnp.ones((n_interior, dim))
  interior_w = jnp.ones((n_interior,))
  boundary_x = jnp.ones((n_boundary, dim))
  boundary_w = jnp.ones((n_boundary,))
  boundary_tangent = jnp.ones((n_boundary, dim))
  boundary_normal = jnp.ones((n_boundary, dim))
  meta = {"shape_test": True, "dim": dim}
  return Quadrature(
    dim=dim,
    interior_x=interior_x,
    interior_w=interior_w,
    boundary_x=boundary_x,
    boundary_w=boundary_w,
    boundary_tangent=boundary_tangent,
    boundary_normal=boundary_normal,
    meta=meta,
  )


def _dummy_state(quad: Quadrature, n_states: int):
  """
  Minimal FunctionState-like payload filled with ones for shape testing.
  Returned object is a simple namespace with the expected attributes.
  """
  Ni, Nb, dim = quad.interior_x.shape[0], quad.boundary_x.shape[0], quad.dim
  class _FS:
    pass
  fs = _FS()
  fs.interior = jnp.ones((Ni, n_states))
  fs.boundary = jnp.ones((Nb, n_states))
  fs.grad_interior = jnp.ones((Ni, n_states, dim))
  fs.grad_boundary = jnp.ones((Nb, n_states, dim))
  fs.n_states = n_states
  return fs

# ---------------------------
# 1D comparison
# ---------------------------

def compare_num_exact_1d(
  X: jax.Array,
  u_num_fn: UFn,
  u_exact_fn: UFn,
  titles: Tuple[str, str, str] = ("Exact", "Numerical", "Error (num - exact)"),
  savepath: Optional[str] = None,
):
  """
  Line plots vs x for 1D: exact / numerical / error.
  """
  Xn = np.asarray(X).reshape(-1, 1)
  assert Xn.shape[1] == 1, f"Expected 1D points, got shape {Xn.shape}"

  x  = Xn[:, 0]
  un = np.asarray(u_num_fn(X)).reshape(-1)
  ue = np.asarray(u_exact_fn(X)).reshape(-1)
  err = un - ue

  idx = np.argsort(x); x, un, ue, err = x[idx], un[idx], ue[idx], err[idx]

  fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
  ax[0].plot(x, ue);  ax[0].set_title(titles[0])
  ax[1].plot(x, un);  ax[1].set_title(titles[1])
  ax[2].plot(x, err); ax[2].set_title(titles[2])

  for a in ax:
    a.set_xlabel("x"); a.set_ylabel("u"); a.grid(True, alpha=0.3)

  if savepath:
    fig.savefig(savepath, dpi=150)
  return fig, ax

# ---------------------------
# 2D helpers
# ---------------------------

def _values_2d(X: jax.Array, u_fn: UFn) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  Xn = np.asarray(X); assert Xn.shape[1] == 2, f"Expected 2D points, got {Xn.shape}"
  x, y = Xn[:, 0], Xn[:, 1]
  val  = np.asarray(u_fn(X)).reshape(-1)
  return x, y, val

def _scatter_panel(ax, x, y, val, title, cmap="viridis"):
  sc = ax.scatter(x, y, c=val, s=12, cmap=cmap)
  ax.set_title(title); ax.set_aspect("equal")
  plt.colorbar(sc, ax=ax)

def _tri_panel(ax, x, y, val, title, cmap="viridis"):
  tri = Triangulation(x, y)
  tpc = ax.tripcolor(tri, val, shading="gouraud", cmap=cmap)
  ax.set_title(title); ax.set_aspect("equal")
  plt.colorbar(tpc, ax=ax)

def compare_num_exact_2d(
  X: jax.Array,
  u_num_fn: UFn,
  u_exact_fn: UFn,
  kind: Literal["scatter","tri"] = "scatter",
  titles: Optional[Tuple[str, str, str]] = None,
  error_kind: Literal["absolute", "relative"] = "absolute",
  savepath: Optional[str] = None,
):
  """
  2D: three panels (exact / numerical / error or relative error).
  kind = 'scatter' (fast, pointwise) or 'tri' (triangulated interpolation).
  error_kind selects the third panel: absolute error (num - exact) or relative
  error (num - exact) / |exact|.
  """
  x, y, un = _values_2d(X, u_num_fn)
  _, _, ue  = _values_2d(X, u_exact_fn)
  err = un - ue
  if error_kind == "relative":
    eps = np.finfo(err.dtype).eps if np.issubdtype(err.dtype, np.floating) else 1e-12
    denom = np.maximum(np.abs(ue), eps)
    err = err / denom
  elif error_kind != "absolute":
    raise ValueError("error_kind must be 'absolute' or 'relative'")

  if titles is None:
    third = "Relative Error" if error_kind == "relative" else "Error (num - exact)"
    titles = ("Exact", "Numerical", third)

  fig, ax = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
  panel = _scatter_panel if kind == "scatter" else _tri_panel
  if kind not in ("scatter", "tri"):
    raise ValueError("kind must be 'scatter' or 'tri'")

  panel(ax[0], x, y, ue,  titles[0], cmap="viridis")
  panel(ax[1], x, y, un,  titles[1], cmap="viridis")
  panel(ax[2], x, y, err, titles[2], cmap="RdBu")

  if savepath:
    fig.savefig(savepath, dpi=150)
  return fig, ax

def plot_numeric_2d(
  X: jax.Array,
  u_num_fn: UFn,
  kind: Literal["scatter","tri"] = "scatter",
  ax: Optional[plt.Axes] = None,
  title: str = "Numerical",
  cmap: str = "viridis",
  add_colorbar: bool = True,
):
  """
  2D numerical-only plotting. Accepts an Axes (like seaborn), or creates one.
  Returns (fig, ax) so you can compose.
  """
  x, y, un = _values_2d(X, u_num_fn)

  created = False
  if ax is None:
    created = True
    fig, ax = plt.subplots(1, 1, figsize=(5.0, 4.5), constrained_layout=True)
  else:
    fig = ax.figure

  if kind == "scatter":
    sc = ax.scatter(x, y, c=un, s=12, cmap=cmap)
    if add_colorbar: plt.colorbar(sc, ax=ax)
  elif kind == "tri":
    tri = Triangulation(x, y)
    tpc = ax.tripcolor(tri, un, shading="gouraud", cmap=cmap)
    if add_colorbar: plt.colorbar(tpc, ax=ax)
  else:
    raise ValueError("kind must be 'scatter' or 'tri'")

  ax.set_title(title); ax.set_aspect("equal")
  return fig, ax
