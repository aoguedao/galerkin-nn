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