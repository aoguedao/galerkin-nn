# %%
import matplotlib.pyplot as plt
import jax.numpy as jnp


from galerkinnn.quadratures import (
  dd_overlapping_interval_quadratures,
  dd_overlapping_rectangle_quadratures,
  dd_overlapping_disk_annulus_quadratures,
  dd_overlapping_rectangle_four_quadratures,
  dd_overlapping_disk_rectangle_quadratures,
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


def plot_dd_four_2d(ax, Qs, title: str):
  """
  Plot four 2D DDQuadratures in one axis with different markers/colors.
  """
  markers = ["o", "s", "^", "v"]
  colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
  labels = ["sub 0", "sub 1", "sub 2", "sub 3"]

  for Q, m, c, label in zip(Qs, markers, colors, labels):
    xi = Q.interior_x[:, 0]
    yi = Q.interior_x[:, 1]
    ax.scatter(xi, yi, s=8, alpha=0.6, marker=m, color=c, label=f"{label} interior")

    xb = Q.boundary_x[:, 0]
    yb = Q.boundary_x[:, 1]
    is_global = Q.boundary_mask_global
    ax.scatter(xb[is_global], yb[is_global], s=20, marker="x", color="black", alpha=0.8)
    ax.scatter(xb[~is_global], yb[~is_global], s=20, marker="D", color=c, alpha=0.8)

  ax.set_aspect("equal", adjustable="box")
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_title(title)
  ax.legend(loc="upper right", fontsize=8, ncol=2)


def test_dd_overlapping_rectangle_four_quadratures():
  bounds = ((0.0, 1.0), (0.0, 1.0))
  Qs = dd_overlapping_rectangle_four_quadratures(
    bounds=bounds,
    midx=0.5,
    midy=0.5,
    overlapx=0.2,
    overlapy=0.2,
    nx=4,
    ny=4,
    n_edge=4,
  )

  assert len(Qs) == 4
  a, b = bounds[0]
  c, d = bounds[1]
  eps = 1e-12

  for idx, Q in enumerate(Qs):
    owner = Q.owner_j_at_bndry
    mask_g = Q.boundary_mask_global
    x_b = Q.boundary_x[:, 0]
    y_b = Q.boundary_x[:, 1]

    assert Q.subdomain_id == idx
    assert len(Q.neighbor_ids) == 3
    assert owner.shape[0] == Q.boundary_x.shape[0]
    assert owner.shape == mask_g.shape
    assert jnp.array_equal(mask_g, owner == -1)

    on_global = (x_b <= a + eps) | (x_b >= b - eps) | (y_b <= c + eps) | (y_b >= d - eps)
    assert jnp.all(owner[on_global] == -1)

    # At least one interface point per subdomain.
    assert jnp.any(~mask_g)

    allowed = (owner == -1)
    for nid in Q.neighbor_ids:
      allowed = allowed | (owner == nid)
    assert jnp.all(allowed)


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

  # ----------- 2D overlapping rectangles: four subdomains -----------
  Qs_rect4 = dd_overlapping_rectangle_four_quadratures(
    bounds=((0.0, 1.0), (0.0, 1.0)),
    midx=0.5,
    midy=0.5,
    overlapx=0.2,
    overlapy=0.2,
    nx=8,
    ny=8,
    n_edge=8,
  )

# ----------- disk + annulus decomposition -----------
  Q_disk, Q_ann = dd_overlapping_disk_annulus_quadratures(
    R=1.0,
    r=0.5,
    overlap=0.1,
    n_r_disk=6,
    n_r_annulus=6,
    n_theta=32,
  )

  # ----------- disk + rectangle decomposition -----------
  Q_disk_dr, Q_rect_dr = dd_overlapping_disk_rectangle_quadratures(
    R=1.0,
    rect_bounds=((0.0, 1.8), (-0.6, 0.6)),  # tweak to match your PDF
    n_r=16,
    n_theta=64,
    nx=16,
    ny=16,
    n_edge=None,
  )

  fig, axes = plt.subplots(5, 1, figsize=(7, 20))

  # Row 1: 1D pair
  plot_dd_pair_1d(axes[0], Q0_1d, Q1_1d, "1D overlapping intervals")

  # Row 2: rectangles
  plot_dd_pair_2d(axes[1], Q0_rect, Q1_rect, "Overlapping rectangles")

  # Row 3: four overlapping rectangles
  plot_dd_four_2d(axes[2], Qs_rect4, "Overlapping rectangles (4 subdomains)")

  # Row 4: disk + annulus
  plot_dd_pair_2d(axes[3], Q_disk, Q_ann, "Disk + annulus decomposition")

  # Row 5: disk + rectangle
  plot_dd_pair_2d(axes[4], Q_disk_dr, Q_rect_dr, "Disk + rectangle DD")

  plt.tight_layout()
  plt.show()

if __name__ == "__main__":
  main()
