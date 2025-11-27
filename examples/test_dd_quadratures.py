# %%
import matplotlib.pyplot as plt
import jax.numpy as jnp


from galerkinnn.quadratures import (
  dd_overlapping_interval_quadratures,
  dd_overlapping_rectangle_quadratures,
  dd_disk_annulus_quadratures,
)
def plot_dd_pair_1d(ax, Q0, Q1, title: str):
  """
  Plot two 1D DDQuadratures in the same axis.
  Q0: subdomain 0, Q1: subdomain 1
  """
  # y=0 reference line
  ax.axhline(0.0, color="black", linewidth=0.5)

  # --- interiors ---
  x0_int = Q0.interior_x[:, 0]
  x1_int = Q1.interior_x[:, 0]

  ax.scatter(x0_int, jnp.zeros_like(x0_int), s=10, label="sub 0 interior")
  ax.scatter(x1_int, jnp.zeros_like(x1_int), s=10, label="sub 1 interior")

  # --- boundaries ---
  def _plot_boundary(Q, label_prefix):
    x_bnd = Q.boundary_x[:, 0]
    is_global = Q.boundary_mask_global
    x_global = x_bnd[is_global]
    x_iface  = x_bnd[~is_global]

    if x_global.size > 0:
      ax.scatter(
        x_global,
        jnp.zeros_like(x_global),
        s=60,
        marker="x",
        color="red",
        label=f"{label_prefix} global"
      )
    if x_iface.size > 0:
      ax.scatter(
        x_iface,
        jnp.zeros_like(x_iface),
        s=60,
        marker="D",
        color="magenta",
        label=f"{label_prefix} interface"
      )

  _plot_boundary(Q0, "sub 0")
  _plot_boundary(Q1, "sub 1")

  ax.set_yticks([])
  ax.set_xlabel("x")
  ax.set_title(title)
  ax.legend(loc="upper right", fontsize=8)


def plot_dd_pair_2d(ax, Q0, Q1, title: str):
  """
  Plot two 2D DDQuadratures in the same axis.
  Q0: subdomain 0, Q1: subdomain 1
  """

  # --- interiors ---
  x0_int = Q0.interior_x[:, 0]
  y0_int = Q0.interior_x[:, 1]
  x1_int = Q1.interior_x[:, 0]
  y1_int = Q1.interior_x[:, 1]

  ax.scatter(x0_int, y0_int, s=8, alpha=0.6, label="sub 0 interior")
  ax.scatter(x1_int, y1_int, s=8, alpha=0.6, label="sub 1 interior")

  # --- boundaries ---
  def _plot_boundary(Q, label_prefix):
    x_bnd = Q.boundary_x[:, 0]
    y_bnd = Q.boundary_x[:, 1]
    is_global = Q.boundary_mask_global
    xg = x_bnd[is_global]
    yg = y_bnd[is_global]
    xi = x_bnd[~is_global]
    yi = y_bnd[~is_global]

    if xg.size > 0:
      ax.scatter(
        xg,
        yg,
        s=25,
        marker="x",
        color="red",
        label=f"{label_prefix} global"
      )
    if xi.size > 0:
      ax.scatter(
        xi,
        yi,
        s=25,
        marker="D",
        color="magenta",
        label=f"{label_prefix} interface"
      )

  _plot_boundary(Q0, "sub 0")
  _plot_boundary(Q1, "sub 1")

  ax.set_aspect("equal", adjustable="box")
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_title(title)
  ax.legend(loc="upper right", fontsize=8)


# =========================
#       Main script
# =========================

def main():
  # ----------- 1D overlapping intervals -----------
  Q0_1d, Q1_1d = dd_overlapping_interval_quadratures(
    bounds=(0.0, 1.0),
    mid=0.5,
    overlap=0.3,
    n_interior=16,
  )

  # ----------- 2D overlapping rectangles -----------
  Q0_rect, Q1_rect = dd_overlapping_rectangle_quadratures(
    bounds=((0.0, 1.0), (0.0, 1.0)),
    mid=0.5,
    overlap=0.3,
    nx=8,
    ny=8,
    n_edge=8,
  )

  # ----------- disk + annulus decomposition -----------
  Q_disk, Q_ann = dd_disk_annulus_quadratures(
    R=1.0,
    r=0.5,
    n_r_disk=6,
    n_r_annulus=6,
    n_theta=32,
  )

  fig, axes = plt.subplots(3, 1, figsize=(7, 12))

  # Row 1: 1D pair
  plot_dd_pair_1d(axes[0], Q0_1d, Q1_1d, "1D overlapping intervals")

  # Row 2: rectangles
  plot_dd_pair_2d(axes[1], Q0_rect, Q1_rect, "Overlapping rectangles")

  # Row 3: disk + annulus
  plot_dd_pair_2d(axes[2], Q_disk, Q_ann, "Disk + annulus decomposition")

  plt.tight_layout()
  plt.show()

main()
