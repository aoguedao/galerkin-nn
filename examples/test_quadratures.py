# %%
import matplotlib.pyplot as plt
import jax.numpy as jnp

from galerkinnn.quadratures import (
  interval_quadrature,
  rectangle_quadrature,
  disk_quadrature,
  annulus_quadrature,
)


def plot_quadrature_1d(ax, quad, title: str):
  """
  1D plot: domain on x-axis, interior vs boundary nodes.
  """
  x_int = quad.interior_x[:, 0]
  x_bnd = quad.boundary_x[:, 0]

  ax.axhline(0.0, color="black", linewidth=0.5)
  ax.scatter(x_int, jnp.zeros_like(x_int), s=10, label="interior")
  ax.scatter(x_bnd, jnp.zeros_like(x_bnd), s=40, marker="x", label="boundary")

  ax.set_title(title)
  ax.set_xlabel("x")
  ax.set_yticks([])
  ax.legend(loc="upper right")


def plot_quadrature_2d(ax, quad, title: str, equal_aspect: bool = True):
  """
  2D scatter: interior vs boundary nodes.
  """
  x_int = quad.interior_x[:, 0]
  y_int = quad.interior_x[:, 1]
  x_bnd = quad.boundary_x[:, 0]
  y_bnd = quad.boundary_x[:, 1]

  ax.scatter(x_int, y_int, s=8, alpha=0.6, label="interior")
  ax.scatter(x_bnd, y_bnd, s=25, marker="x", label="boundary")

  ax.set_title(title)
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  if equal_aspect:
    ax.set_aspect("equal", adjustable="box")
  ax.legend(loc="upper right")


def main():
  # --- build quadratures ---
  # 1D interval [0,1]
  q_interval = interval_quadrature(bounds=(0.0, 1.0), n_interior=16)

  # 2D rectangle [0,1]x[0,1]
  q_rect = rectangle_quadrature(((0.0, 1.0), (0.0, 1.0)), nx=8, ny=8, n_edge=8)

  # disk of radius 1
  q_disk = disk_quadrature(radius=1.0, n_r=8, n_theta=32)

  # annulus r in [0.5, 1.0]
  q_ann = annulus_quadrature(r_inner=0.5, r_outer=1.0, n_r=8, n_theta=32)

  # --- plotting ---
  fig, axes = plt.subplots(2, 2, figsize=(10, 10))

  plot_quadrature_1d(axes[0, 0], q_interval, "Interval quadrature [0,1]")
  plot_quadrature_2d(axes[0, 1], q_rect, "Rectangle quadrature [0,1]×[0,1]")
  plot_quadrature_2d(axes[1, 0], q_disk, "Disk quadrature (R=1)")
  plot_quadrature_2d(axes[1, 1], q_ann, "Annulus quadrature (0.5 ≤ r ≤ 1)", equal_aspect=True)

  plt.tight_layout()
  plt.show()


main()
