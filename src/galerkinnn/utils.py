import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from typing import Callable, Optional, Tuple, Literal, Sequence

UFn   = Callable[[jax.Array], jax.Array]  # X -> (N,1)
SigmaFn = Callable[..., jax.Array]  # expects fn(X=...) -> (N, m_i) or (N,1)

# ---------------------------
# Reconstruct u(X) callable
# ---------------------------

def make_u_fn(
  sigma_net_fn_list: Sequence[SigmaFn],
  u_coeff: jax.Array,                       # (B,1)
  basis_coeff_list: Optional[Sequence[jax.Array]] = None  # each (m_i,1) if nets are vector-valued
) -> Callable[[jax.Array], jax.Array]:
  """
  Builds u(X). If basis_coeff_list is given (or fn outputs are not scalar),
  we project each sigma-net output (N,m_i) -> (N,1) using its coeff.
  """
  B = len(sigma_net_fn_list)
  if u_coeff.ndim == 1:
    u_coeff = u_coeff[:, None]          # ensure (B,1)

  def u_fn(X: jax.Array) -> jax.Array:
    cols = []
    for i, fn in enumerate(sigma_net_fn_list):
      out = fn(X=X)                     # (N,m_i) or (N,1)
      if out.ndim != 2 or out.shape[0] != X.shape[0]:
        raise ValueError(f"sigma_net_fn[{i}] returned shape {out.shape}, expected (N,m).")

      # If already scalar, keep it. Otherwise project with basis_coeff_list[i].
      if out.shape[1] == 1:
        phi_i = out                     # (N,1)
      else:
        if basis_coeff_list is None:
          raise ValueError(
            f"sigma_net_fn[{i}] returned (N,{out.shape[1]}), but no basis_coeff_list given."
          )
        coeff_i = basis_coeff_list[i]
        if coeff_i.ndim == 1: coeff_i = coeff_i[:, None]   # (m_i,1)
        phi_i = out @ coeff_i                               # (N,1)
      cols.append(phi_i)

    Phi = jnp.concatenate(cols, axis=1)   # (N,B)
    return Phi @ u_coeff                  # (N,1)

  return u_fn

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
  titles: Tuple[str, str, str] = ("Exact", "Numerical", "Error (num - exact)"),
  savepath: Optional[str] = None,
):
  """
  2D: three panels (exact / numerical / error).
  kind = 'scatter' (fast, pointwise) or 'tri' (triangulated interpolation).
  """
  x, y, un = _values_2d(X, u_num_fn)
  _, _, ue  = _values_2d(X, u_exact_fn)
  err = un - ue

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
