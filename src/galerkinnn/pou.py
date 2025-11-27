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
