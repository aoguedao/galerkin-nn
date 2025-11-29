import jax
import jax.numpy as jnp

def build_pou_weights_1d(Q0, Q1):
  a0, b0 = float(Q0.boundary_x[0,0]), float(Q0.boundary_x[-1,0])
  a1, b1 = float(Q1.boundary_x[0,0]), float(Q1.boundary_x[-1,0])
  a_ov, b_ov = max(a0, a1), min(b0, b1)
  L = max(b_ov - a_ov, 1e-16)

  def w0_fn(X):
    x = jnp.asarray(X).reshape(-1)
    in0 = (x >= a0) & (x <= b0)
    in1 = (x >= a1) & (x <= b1)
    on_ov = in0 & in1
    w = jnp.where(in0 & ~in1, 1.0, 0.0)
    w = jnp.where(on_ov, (b_ov - x) / L, w)
    return w[:, None]

  def w1_fn(X):
    x = jnp.asarray(X).reshape(-1)
    in0 = (x >= a0) & (x <= b0)
    in1 = (x >= a1) & (x <= b1)
    on_ov = in0 & in1
    w = jnp.where(in1 & ~in0, 1.0, 0.0)
    w = jnp.where(on_ov, (x - a_ov) / L, w)
    return w[:, None]

  return w0_fn, w1_fn


def build_pou_weights_rect(Q0, Q1):
  """Partition-of-unity weights along the x-direction overlap."""
  x0 = jnp.asarray(Q0.boundary_x)[:, 0]
  x1 = jnp.asarray(Q1.boundary_x)[:, 0]
  x0_min, x0_max = float(x0.min()), float(x0.max())
  x1_min, x1_max = float(x1.min()), float(x1.max())
  a_ov, b_ov = max(x0_min, x1_min), min(x0_max, x1_max)
  L = max(b_ov - a_ov, 1e-12)

  def w0_fn(X):
    X = jnp.asarray(X).reshape(-1, 2)
    x = X[:, 0]
    in0 = (x >= x0_min) & (x <= x0_max)
    in1 = (x >= x1_min) & (x <= x1_max)
    on_ov = in0 & in1
    w = jnp.where(in0 & ~in1, 1.0, 0.0)
    w = jnp.where(on_ov, (b_ov - x) / L, w)
    return w.reshape(-1, 1)

  def w1_fn(X):
    X = jnp.asarray(X).reshape(-1, 2)
    x = X[:, 0]
    in0 = (x >= x0_min) & (x <= x0_max)
    in1 = (x >= x1_min) & (x <= x1_max)
    on_ov = in0 & in1
    w = jnp.where(in1 & ~in0, 1.0, 0.0)
    w = jnp.where(on_ov, (x - a_ov) / L, w)
    return w.reshape(-1, 1)

  return w0_fn, w1_fn


def build_pou_weights_disk_smooth(Q0, Q1):
  """
  Smooth radial partition of unity for disk + annulus, based on the actual
  radial supports of the two DDQuadratures.

  Let [r0_min, r0_max] be the radial support of Ω0,
      [r1_min, r1_max] the radial support of Ω1.
  The overlap is [a_ov, b_ov] = [max(r0_min, r1_min), min(r0_max, r1_max)].

  Define w0(r) = 1 on r <= a_ov,
                 smooth cosine ramp ↓ on [a_ov, b_ov],
                 0 on r >= b_ov,
         w1 = 1 - w0.
  """

  X0 = jnp.asarray(Q0.interior_x)
  X1 = jnp.asarray(Q1.interior_x)

  r0 = jnp.sqrt(jnp.sum(X0**2, axis=1))
  r1 = jnp.sqrt(jnp.sum(X1**2, axis=1))

  r0_min, r0_max = float(r0.min()), float(r0.max())
  r1_min, r1_max = float(r1.min()), float(r1.max())

  a_ov = max(r0_min, r1_min)
  b_ov = min(r0_max, r1_max)
  L = max(b_ov - a_ov, 1e-12)

  def w0_fn(X):
    X = jnp.asarray(X).reshape(-1, 2)
    r = jnp.sqrt(jnp.sum(X**2, axis=1))

    # Regions
    left  = r <= a_ov
    right = r >= b_ov
    mid   = (~left) & (~right)

    w = jnp.zeros_like(r)

    # 1 in inner-only region
    w = jnp.where(left, 1.0, w)

    # smooth cosine on overlap
    t = (r - a_ov) / L
    t = jnp.clip(t, 0.0, 1.0)
    w_mid = 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    w = jnp.where(mid, w_mid, w)

    # 0 in outer-only region is already the default
    return w.reshape(-1, 1)

  def w1_fn(X):
    return 1.0 - w0_fn(X)

  return w0_fn, w1_fn


def build_pou_weights_disk_rect(Q_disk, Q_rect):
  """
  Simple partition of unity for disk + rectangle overlap.

  Geometry (from DDQuadrature meta):
    - Q_disk:  Ω0 = { (x,y): x^2 + y^2 <= R^2 }.
    - Q_rect:  Ω1 = [ax,bx] × [ay,by].

  We define:

    in_disk  = indicator of Ω0
    in_rect  = indicator of Ω1
    in_union = in_disk ∪ in_rect
    both     = in_disk ∩ in_rect

  Weights:

    - disk-only:      w_disk = 1, w_rect = 0
    - rect-only:      w_disk = 0, w_rect = 1
    - overlap (both): w_disk = 0.5, w_rect = 0.5
    - outside union:  w_disk = 0, w_rect = 0  (unused)

  So on Ω we have w_disk + w_rect = 1 and both weights lie in [0,1].
  """

  meta0 = Q_disk.meta or {}
  meta1 = Q_rect.meta or {}

  # Disk radius
  if "R_global" in meta0:
    R = float(meta0["R_global"])
  else:
    Xd = jnp.asarray(Q_disk.interior_x)
    r = jnp.sqrt(jnp.sum(Xd**2, axis=1))
    R = float(r.max())

  # Rectangle bounds
  if "rect_bounds" in meta1:
    (ax, bx), (ay, by) = meta1["rect_bounds"]
    ax, bx, ay, by = float(ax), float(bx), float(ay), float(by)
  else:
    Xr = jnp.asarray(Q_rect.interior_x)
    xr = Xr[:, 0]
    yr = Xr[:, 1]
    ax, bx = float(xr.min()), float(xr.max())
    ay, by = float(yr.min()), float(yr.max())

  def _membership(X):
    X = jnp.asarray(X).reshape(-1, 2)
    x = X[:, 0:1]
    y = X[:, 1:2]

    r = jnp.sqrt(jnp.sum(X**2, axis=1, keepdims=True))
    in_disk = r <= (R + 1e-8)
    in_rect = (x >= ax) & (x <= bx) & (y >= ay) & (y <= by)
    in_union = in_disk | in_rect
    only_disk = in_disk & (~in_rect)
    only_rect = in_rect & (~in_disk)
    both = in_disk & in_rect
    return in_disk, in_rect, in_union, only_disk, only_rect, both

  def w_disk_fn(X):
    in_disk, in_rect, in_union, only_disk, only_rect, both = _membership(X)
    x = jnp.asarray(X).reshape(-1, 2)[:, 0:1]  # shape helper

    w0 = jnp.zeros_like(x)

    # disk-only
    w0 = jnp.where(only_disk, 1.0, w0)

    # overlap
    w0 = jnp.where(both, 0.5, w0)

    # outside union: keep 0
    w0 = jnp.where(in_union, w0, 0.0)

    return w0  # (N,1)

  def w_rect_fn(X):
    in_disk, in_rect, in_union, only_disk, only_rect, both = _membership(X)
    x = jnp.asarray(X).reshape(-1, 2)[:, 0:1]

    w1 = jnp.zeros_like(x)

    # rect-only
    w1 = jnp.where(only_rect, 1.0, w1)

    # overlap
    w1 = jnp.where(both, 0.5, w1)

    # outside union: keep 0
    w1 = jnp.where(in_union, w1, 0.0)

    return w1  # (N,1)

  return w_disk_fn, w_rect_fn
