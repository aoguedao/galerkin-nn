# Domain decomposition (Robin–Schwarz ASM) for 1D Poisson with piecewise k and physical Robin BCs.
# %%
import jax
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from flax import struct
from typing import Callable, Tuple, Optional

from galerkinnn import FunctionState, PDE, Quadrature, GalerkinNNDD
from galerkinnn.quadratures import dd_overlapping_interval_quadratures
from galerkinnn.formulations import DDPDE
from galerkinnn.utils import make_u_fn, make_impedance_trace

# ---------------- Hyper-parameters ----------------
seed = 42
max_bases = 5
max_epoch_basis = 50
tol_solution = 1e-7
tol_basis = 1e-7

n_train = 128             # quadrature points per subdomain
N = 5                     # init neurons
r = 2                     # neurons growth per basis
A = 5e-3                  # init learning rate
rho = 1.1                 # LR decay per basis

# problem params
bounds = (0.0, 1.0)
mid = 0.5
overlap = 0.2
k1 = 1.0
k2 = 50.0
x0 = 0.5
eps = 1e-3          # physical Robin parameter (u + eps * (±k u') = 0 on boundary)
a_curv = 1.0        # second-derivative = 2*a_curv in each piece

# ===== NN definitions =====
def net_fn(X, params, activation):
  X = jnp.dot(X, params["W"]) + params["b"]
  return activation(X)

def activations_fn(i):
  s = i
  return lambda x: jnp.tanh(s * x)

network_widths_fn = lambda i: N * (r ** (i - 1))
learning_rates_fn = lambda i: A * (rho ** (-(i - 1)))

# ===== PDE definition (same as before, now with .with_k) =====
@struct.dataclass
class Poisson1DJumpRobin(PDE):
  k1: float = 1.0
  k2: float = 10.0
  x0: float = 0.5
  eps: float = 1e-3
  a_curv: float = 1.0
  kappa_fn: Optional[Callable[[jax.Array], jax.Array]] = struct.field(
    pytree_node=False, default=None
  )

  def k(self, X: jax.Array) -> jax.Array:
    if self.kappa_fn is not None:
      kx = self.kappa_fn(X)
      return kx if kx.ndim == 2 else kx.reshape(-1, 1)
    x = X.reshape(-1)
    return jnp.where(x <= self.x0, self.k1, self.k2)[:, None]

  def with_k(self, kappa_fn: Callable[[jax.Array], jax.Array]) -> "Poisson1DJumpRobin":
    return self.replace(kappa_fn=kappa_fn)

  def source(self) -> Callable[[jax.Array], jax.Array]:
    a = self.a_curv
    def f(X: jax.Array) -> jax.Array:
      kx = self.k(X)
      return -2.0 * a * kx
    return f

  def bilinear_form(self):
    eps = self.eps
    def a(u: FunctionState, v: FunctionState, quad: Quadrature) -> jax.Array:
      kx = self.k(quad.interior_x).reshape(-1)
      a_vol = jnp.einsum(
        "nui,nvi,n->uv", u.grad_interior, v.grad_interior, quad.interior_w * kx
      )
      gamma_b = (1.0 / eps) * quad.boundary_w
      mask_g = getattr(quad, "boundary_mask_global", None)
      if mask_g is not None:
        gamma_b = gamma_b * mask_g.astype(gamma_b.dtype)
      a_bnd = jnp.einsum("an,am,a->nm", u.boundary, v.boundary, gamma_b)
      return a_vol + a_bnd
    return a

  def linear_operator(self):
    f = self.source()
    def L(v: FunctionState, quad: Quadrature) -> jnp.ndarray:
      fvals = f(quad.interior_x).reshape(-1)
      return jnp.einsum("n,nv,n->v", fvals, v.interior, quad.interior_w)
    return L

  def energy_norm(self):
    eps = self.eps
    def norm(v: FunctionState, quad: Quadrature) -> jax.Array:
      kx = self.k(quad.interior_x).reshape(-1)
      grad_sq = jnp.sum(v.grad_interior**2, axis=2)
      a1 = jnp.einsum("n,ni->i", quad.interior_w * kx, grad_sq)
      b_sq = v.boundary**2
      gamma_b = (1.0 / eps) * quad.boundary_w
      mask_g = getattr(quad, "boundary_mask_global", None)
      if mask_g is not None:
        gamma_b = gamma_b * mask_g.astype(gamma_b.dtype)
      a2 = jnp.einsum("n,ni->i", gamma_b, b_sq)
      en2 = jnp.maximum(a1 + a2, jnp.array(0.0, a1.dtype))
      return jnp.sqrt(en2)
    return norm

# =========================== Domain setup ===========================
Q0, Q1 = dd_overlapping_interval_quadratures(bounds=bounds, mid=mid, overlap=overlap, ng=n_train)
print("n0_R =", float(Q0.boundary_normal[-1,0]), "  n1_L =", float(Q1.boundary_normal[0,0]))
print("Ω0 boundary_mask_global:", np.array(Q0.boundary_mask_global))
print("Ω1 boundary_mask_global:", np.array(Q1.boundary_mask_global))

print("Q0.neighbor_ids:", Q0.neighbor_ids)
print("Q1.neighbor_ids:", Q1.neighbor_ids)
print("Q0.onehot shape:", Q0.boundary_owner_onehot.shape)
print("Q1.onehot shape:", Q1.boundary_owner_onehot.shape)

pde_template = Poisson1DJumpRobin(k1=k1, k2=k2, x0=x0, eps=eps, a_curv=a_curv)
base_pde0 = pde_template.with_k(lambda X: jnp.full((X.shape[0], 1), k1))
base_pde1 = pde_template.with_k(lambda X: jnp.full((X.shape[0], 1), k2))

# %%
# =========================== Run GalerkinNNDD ===========================
gnndd = GalerkinNNDD(
  base_pde=[base_pde0, base_pde1],
  quadratures=[Q0, Q1],
  eps_interface=1e-6,            # δ (interface Robin parameter)
  transmission="impedance",      # use impedance (Robin) transmission
  trace_relaxation=0.9                    # relaxation
)

asm_out = gnndd.solve(
  net_fn=net_fn,
  activations_fn=activations_fn,
  network_widths_fn=network_widths_fn,
  learning_rates_fn=learning_rates_fn,
  max_bases=max_bases,
  max_epoch_basis=max_epoch_basis,
  tol_solution=tol_solution,
  tol_basis=tol_basis,
  max_sweeps=8,
  tol_jump=5e-5,
  seeds=[seed, seed + 1],
)

def robin_residual(u_fn, Q, base_pde, delta):
  # pick the interface node(s)
  # Ω0: rightmost; Ω1: leftmost
  Xb = Q.boundary_x
  # detect interface rows (mask_global=False)
  mask_g = getattr(Q, "boundary_mask_global", None)
  if mask_g is None:
    # fallback: in 1D, physical boundaries are the two ends of global domain,
    # so interface is the opposite end for each subdomain. Keep last/first row.
    pass
  else:
    Xb = Xb[~mask_g.reshape(-1)]

  if Xb.shape[0] == 0:  # safety
    Xb = Q.boundary_x[-1:,:]

  # outward normal from *target* geometry
  a_t = float(Q.boundary_x[0,0]); b_t = float(Q.boundary_x[-1,0])
  def n_from_X(X):
    x = X.reshape(-1)
    is_left  = jnp.isclose(x, a_t, atol=1e-12)
    is_right = jnp.isclose(x, b_t, atol=1e-12)
    return (is_right.astype(X.dtype) - is_left.astype(X.dtype)).reshape(-1, 1)

  def u_scalar(x): return u_fn(x[None,:]).reshape(())
  du = jax.vmap(jax.grad(u_scalar))(Xb).reshape(Xb.shape[0], 1)  # (Nb,1)
  n  = n_from_X(Xb)                                              # (Nb,1)
  kx = base_pde.k(Xb).reshape(-1,1)                              # TARGET κ

  # Residual: r = κ ∂_n u + (1/δ) u − (1/δ) g, with g built from the *other* side.
  # Compute g using the same builder to avoid mismatch:
  # For diagnostics we need some neighbor u_fn_other; pass it in when calling.
  return lambda u_fn_other: (
    kx * du * n + (1.0/delta) * u_fn(Xb) - (1.0/delta) *
    make_impedance_trace(u_fn_other, Q, base_pde.k, delta)(Xb)
  ).reshape(-1)

# get u’s
u0_fn, u1_fn = asm_out["u_fns"]

r0 = robin_residual(u0_fn, Q0, base_pde0, gnndd.eps_interface)(u1_fn)
r1 = robin_residual(u1_fn, Q1, base_pde1, gnndd.eps_interface)(u0_fn)
print("Robin residuals on Γ: Ω0 side:", np.array(r0), "  Ω1 side:", np.array(r1))


# =========================== Plot stitched vs analytic ===========================
# %%
def exact_coeffs(k1: float, k2: float, x0: float, eps: float, a: float) -> Tuple[float,float,float,float]:
  denom = k2 * (eps * k1 + x0) + k1 * (1.0 - x0 + eps * k2)
  num   = k2 * (1.0 + 2.0 * eps * k2) - 2.0 * x0 * (k2 - k1) * (1.0 - x0 + eps * k2)
  bL = -a * (num / denom)
  bR = (k1 * bL - 2.0 * a * x0 * (k2 - k1)) / k2
  cL = eps * k1 * bL
  cR = cL + (bL - bR) * x0
  return float(bL), float(cL), float(bR), float(cR)

def u_exact_fn(k1: float, k2: float, x0: float, eps: float, a: float) -> Callable[[jax.Array], jax.Array]:
  bL, cL, bR, cR = exact_coeffs(k1, k2, x0, eps, a)
  def u(X: jax.Array) -> jax.Array:
    x = X.reshape(-1, 1)
    uL = a * x**2 + bL * x + cL
    uR = a * x**2 + bR * x + cR
    return jnp.where(x <= x0, uL, uR)
  return u

def _eval_u(u_fn, x1d):
  X = jnp.array(x1d, dtype=jnp.float32).reshape(-1, 1)
  val = jnp.asarray(u_fn(X))
  if val.ndim == 2 and val.shape[1] == 1:
    val = val[:, 0]
  return np.array(val)

def plot_asm_solution_with_exact(asm_out, Q0, Q1, pde, n_plot=800,
                                 title="Poisson 1D (physical Robin) — Robin–Schwarz ASM vs analytic"):
  u_fns = asm_out["u_fns"]
  u0_fn, u1_fn = u_fns[0], u_fns[1]

  a0, b0 = float(Q0.boundary_x[0,0]), float(Q0.boundary_x[-1,0])
  a1, b1 = float(Q1.boundary_x[0,0]), float(Q1.boundary_x[-1,0])
  a_ov, b_ov = max(a0, a1), min(b0, b1)
  has_ov = b_ov > a_ov + 1e-14

  x0g = np.linspace(a0, b0, max(200, n_plot//2))
  x1g = np.linspace(a1, b1, max(200, n_plot//2))
  xF = np.linspace(min(a0, a1), max(b0, b1), n_plot)

  u0 = _eval_u(u0_fn, x0g)
  u1 = _eval_u(u1_fn, x1g)

  u_stitch = np.empty_like(xF)
  in0 = (xF >= a0) & (xF <= b0)
  in1 = (xF >= a1) & (xF <= b1)

  uF0 = np.full_like(xF, np.nan, dtype=float)
  uF1 = np.full_like(xF, np.nan, dtype=float)
  if in0.any(): uF0[in0] = _eval_u(u0_fn, xF[in0])
  if in1.any(): uF1[in1] = _eval_u(u1_fn, xF[in1])

  only0 = in0 & ~in1; only1 = in1 & ~in0
  u_stitch[only0] = uF0[only0]
  u_stitch[only1] = uF1[only1]
  if has_ov:
    on_ov = in0 & in1
    w0 = (b_ov - xF[on_ov]) / max(b_ov - a_ov, 1e-16)
    w1 = 1.0 - w0
    u_stitch[on_ov] = w0 * uF0[on_ov] + w1 * uF1[on_ov]

  u_exact = u_exact_fn(k1=pde.k1, k2=pde.k2, x0=pde.x0, eps=pde.eps, a=pde.a_curv)
  u_ex = _eval_u(u_exact, xF)

  x_if_R = float(Q0.boundary_x[-1,0])
  x_if_L = float(Q1.boundary_x[ 0,0])
  u0_if = float(_eval_u(u0_fn, [x_if_R])[0])
  u1_if = float(_eval_u(u1_fn, [x_if_L])[0])
  jump  = u0_if - u1_if

  fig, ax = plt.subplots(figsize=(9, 3.6))
  if has_ov: ax.axvspan(a_ov, b_ov, color="gold", alpha=0.20, label="overlap")
  ax.plot(xF, u_ex, lw=2.5, label="analytic", color="C1")
  ax.plot(xF, u_stitch, lw=2.2, label="DD (stitched)", color="C0")
  ax.plot([x_if_R], [u0_if], "o", ms=6, label="u₀(b₀⁺)")
  ax.plot([x_if_L], [u1_if], "s", ms=6, label="u₁(a₁⁻)")
  ax.annotate(f"jump ≈ {jump:+.3e}", xy=((x_if_R+x_if_L)/2, (u0_if+u1_if)/2),
              xytext=(10, 12), textcoords="offset points", fontsize=9)

  ax.set_xlabel("x"); ax.set_ylabel("u(x)")
  ax.set_title(title)
  ax.legend(ncols=3, fontsize=9, loc="best")
  ax.grid(True, alpha=0.25)
  plt.tight_layout(); plt.show()

  l2 = np.sqrt(np.trapezoid((u_stitch - u_ex)**2, xF))
  print(f"L2 error on plot grid ≈ {l2:.3e}")

plot_asm_solution_with_exact(asm_out, Q0, Q1, pde_template)



# # Domain decomposition (Robin–Schwarz ASM) for 1D Poisson with piecewise k and physical Robin BCs.
# # %%
# import jax
# import matplotlib.pyplot as plt
# import numpy as np
# import jax.numpy as jnp

# from flax import struct
# from typing import Callable, Tuple, Optional

# from galerkinnn import FunctionState, PDE, Quadrature, GalerkinNNDD
# from galerkinnn.quadratures import dd_overlapping_interval_quadratures
# from galerkinnn.formulations import DDPDE
# from galerkinnn.pou import build_pou_weights_1d
# from galerkinnn.utils import make_u_fn, make_impedance_trace

# # ---------------- Hyper-parameters ----------------
# seed = 42
# max_bases = 5
# max_epoch_basis = 50
# tol_solution = 1e-7
# tol_basis = 1e-7

# n_train = 128             # quadrature points per subdomain
# N = 5                     # init neurons
# r = 2                     # neurons growth per basis
# A = 5e-3                  # init learning rate
# rho = 1.1                 # LR decay per basis

# # problem params
# bounds = (0.0, 1.0)
# mid = 0.5
# overlap = 0.2
# k1 = 1.0
# k2 = 50.0
# x0 = 0.5
# eps = 1e-3          # physical Robin parameter (u + eps * (±k u') = 0 on boundary)
# a_curv = 1.0        # second-derivative = 2*a_curv in each piece

# # ===== NN =====
# def net_fn(X, params, activation):
#   X = jnp.dot(X, params["W"]) + params["b"]
#   return activation(X)

# def activations_fn(i):
#   s = i
#   return lambda x: jnp.tanh(s * x)

# network_widths_fn = lambda i: N * (r ** (i - 1))
# learning_rates_fn = lambda i: A * (rho ** (-(i - 1)))

# # ===== PDE =====
# @struct.dataclass
# class Poisson1DJumpRobin(PDE):
#   """
#   -(k u')' = f in (0,1), homogeneous Robin via 1/eps on the physical boundary:
#     a(u,v) = ∫ k u' v' dx + (1/eps) * sum_{b ∈ ∂Ω} u(b) v(b)
#     L(v)   = ∫ f v dx

#   k is piecewise-constant with a jump at x0. Exact u is piecewise quadratic
#   with continuity of u and flux k u' at x0, and Robin on x=0,1.
#   """
#   k1: float = 1.0
#   k2: float = 10.0
#   x0: float = 0.5
#   eps: float = 1e-3
#   a_curv: float = 1.0  # common quadratic coefficient 'a' (u''=2a)

#   # Optional conductivity override
#   kappa_fn: Optional[Callable[[jax.Array], jax.Array]] = struct.field(
#     pytree_node=False, default=None
#   )

#   # ----------------------------------------
#   # Conductivity field
#   # ----------------------------------------
#   def k(self, X: jax.Array) -> jax.Array:
#     """Return κ(X). Uses override if provided, otherwise piecewise default."""
#     if self.kappa_fn is not None:
#       kx = self.kappa_fn(X)
#       return kx if kx.ndim == 2 else kx.reshape(-1, 1)

#     x = X.reshape(-1)
#     return jnp.where(x <= self.x0, self.k1, self.k2)[:, None]  # (N,1)

#   def with_k(self, kappa_fn: Callable[[jax.Array], jax.Array]) -> "Poisson1DJumpRobin":
#     """Return a copy of this PDE with the specified conductivity function."""
#     return self.replace(kappa_fn=kappa_fn)

#   # ----------------------------------------
#   # Source term
#   # ----------------------------------------
#   def source(self) -> Callable[[jax.Array], jax.Array]:
#     """
#     Choose f so that u'' = 2*a_curv in each piece. Since k is constant per piece,
#     f = -(k u'') = -2*a_curv * k(x).
#     """
#     a = self.a_curv

#     def f(X: jax.Array) -> jax.Array:
#       kx = self.k(X)  # (N,1)
#       return -2.0 * a * kx

#     return f

#   # ----------------------------------------
#   # Bilinear form
#   # ----------------------------------------
#   def bilinear_form(self):
#     eps = self.eps

#     def a(u: FunctionState, v: FunctionState, quad: Quadrature) -> jax.Array:
#       # interior: ∫ k u' v' dx
#       kx = self.k(quad.interior_x).reshape(-1)  # (N,)
#       a_vol = jnp.einsum(
#         "nui,nvi,n->uv",
#         u.grad_interior,
#         v.grad_interior,
#         quad.interior_w * kx,
#       )

#       # physical boundary: (1/eps) * Σ_b u(b) v(b)
#       gamma_b = (1.0 / eps) * quad.boundary_w  # (Nb,)
#       mask_g = getattr(quad, "boundary_mask_global", None)
#       if mask_g is not None:
#         gamma_b = gamma_b * mask_g.astype(gamma_b.dtype)

#       a_bnd = jnp.einsum("an,am,a->nm", u.boundary, v.boundary, gamma_b)
#       return a_vol + a_bnd

#     return a

#   # ----------------------------------------
#   # Linear functional
#   # ----------------------------------------
#   def linear_operator(self):
#     f = self.source()

#     def L(v: FunctionState, quad: Quadrature) -> jnp.ndarray:
#       fvals = f(quad.interior_x).reshape(-1)  # (N,)
#       return jnp.einsum("n,nv,n->v", fvals, v.interior, quad.interior_w)

#     return L

#   # ----------------------------------------
#   # Energy norm
#   # ----------------------------------------
#   def energy_norm(self):
#     eps = self.eps

#     def norm(v: FunctionState, quad: Quadrature) -> jax.Array:
#       kx = self.k(quad.interior_x).reshape(-1)  # (N,)
#       grad_sq = jnp.sum(v.grad_interior**2, axis=2)  # (N, n_v)
#       a1 = jnp.einsum("n,ni->i", quad.interior_w * kx, grad_sq)

#       b_sq = v.boundary**2  # (Nb, n_v)
#       gamma_b = (1.0 / eps) * quad.boundary_w
#       mask_g = getattr(quad, "boundary_mask_global", None)
#       if mask_g is not None:
#         gamma_b = gamma_b * mask_g.astype(gamma_b.dtype)

#       a2 = jnp.einsum("n,ni->i", gamma_b, b_sq)
#       en2 = jnp.maximum(a1 + a2, jnp.array(0.0, a1.dtype))
#       return jnp.sqrt(en2)

#     return norm

# # =========================== Run ASM on two subs ==============================
# Q0, Q1 = dd_overlapping_interval_quadratures(bounds=bounds, mid=mid, overlap=overlap, ng=n_train)
# # pde = Poisson1DJumpRobin(k1=k1, k2=k2, x0=x0, eps=eps, a_curv=a_curv)
# pde_template = Poisson1DJumpRobin(k1=1.0, k2=10.0, x0=0.5, eps=1e-3, a_curv=1.0)
# base_pde0 = pde_template.with_k(lambda X: jnp.full((X.shape[0], 1), pde_template.k1))
# base_pde1 = pde_template.with_k(lambda X: jnp.full((X.shape[0], 1), pde_template.k2))

# gnndd = GalerkinNNDD(

# )
# # ========================= Plot stitched vs analytic ==========================
# # %%
# def exact_coeffs(k1: float, k2: float, x0: float, eps: float, a: float) -> Tuple[float,float,float,float]:
#   """
#   u_-(x)=a x^2 + bL x + cL on [0,x0], u_+(x)=a x^2 + bR x + cR on [x0,1].
#   Conditions:
#     (i) u - eps*k1*u'(0) = 0 at x=0   (physical Robin with outward normal -1)
#     (ii) u + eps*k2*u'(1) = 0 at x=1  (outward normal +1)
#     (iii) continuity: u_-(x0) = u_+(x0)
#     (iv) flux cont.:  k1 u_-'(x0) = k2 u_+'(x0)
#   Returns (bL, cL, bR, cR).
#   """
#   denom = k2 * (eps * k1 + x0) + k1 * (1.0 - x0 + eps * k2)
#   num   = k2 * (1.0 + 2.0 * eps * k2) - 2.0 * x0 * (k2 - k1) * (1.0 - x0 + eps * k2)
#   bL = -a * (num / denom)
#   bR = (k1 * bL - 2.0 * a * x0 * (k2 - k1)) / k2
#   cL = eps * k1 * bL
#   cR = cL + (bL - bR) * x0
#   return float(bL), float(cL), float(bR), float(cR)

# def u_exact_fn(k1: float, k2: float, x0: float, eps: float, a: float) -> Callable[[jax.Array], jax.Array]:
#   bL, cL, bR, cR = exact_coeffs(k1, k2, x0, eps, a)
#   def u(X: jax.Array) -> jax.Array:
#     x = X.reshape(-1, 1)
#     uL = a * x**2 + bL * x + cL
#     uR = a * x**2 + bR * x + cR
#     return jnp.where(x <= x0, uL, uR)
#   return u

# def _eval_u(u_fn, x1d):
#   X = jnp.array(x1d, dtype=jnp.float32).reshape(-1, 1)
#   val = jnp.asarray(u_fn(X))
#   if val.ndim == 2 and val.shape[1] == 1:
#     val = val[:, 0]
#   return np.array(val)

# def plot_asm_solution_with_exact(asm_out, Q0, Q1, pde, n_plot=800,
#                                  title="Poisson 1D (physical Robin) — Robin–Schwarz ASM vs analytic"):
#   u0_fn = asm_out["u0_fn"]; u1_fn = asm_out["u1_fn"]

#   # subdomain bounds
#   a0, b0 = float(Q0.boundary_x[0,0]), float(Q0.boundary_x[-1,0])
#   a1, b1 = float(Q1.boundary_x[0,0]), float(Q1.boundary_x[-1,0])

#   # overlap
#   a_ov, b_ov = max(a0, a1), min(b0, b1)
#   has_ov = b_ov > a_ov + 1e-14

#   # dense grids
#   x0g = np.linspace(a0, b0, max(200, n_plot//2))
#   x1g = np.linspace(a1, b1, max(200, n_plot//2))
#   xF = np.linspace(min(a0, a1), max(b0, b1), n_plot)

#   u0 = _eval_u(u0_fn, x0g)
#   u1 = _eval_u(u1_fn, x1g)

#   # stitched curve (avoid NaNs by only sampling where defined)
#   u_stitch = np.empty_like(xF)
#   in0 = (xF >= a0) & (xF <= b0)
#   in1 = (xF >= a1) & (xF <= b1)

#   uF0 = np.full_like(xF, np.nan, dtype=float)
#   uF1 = np.full_like(xF, np.nan, dtype=float)
#   if in0.any(): uF0[in0] = _eval_u(u0_fn, xF[in0])
#   if in1.any(): uF1[in1] = _eval_u(u1_fn, xF[in1])

#   only0 = in0 & ~in1; only1 = in1 & ~in0
#   u_stitch[only0] = uF0[only0]
#   u_stitch[only1] = uF1[only1]
#   if has_ov:
#     on_ov = in0 & in1
#     w0 = (b_ov - xF[on_ov]) / max(b_ov - a_ov, 1e-16)
#     w1 = 1.0 - w0
#     u_stitch[on_ov] = w0 * uF0[on_ov] + w1 * uF1[on_ov]

#   # analytic solution
#   u_exact = u_exact_fn(k1=pde.k1, k2=pde.k2, x0=pde.x0, eps=pde.eps, a=pde.a_curv)
#   u_ex = _eval_u(u_exact, xF)

#   # interface values for annotation
#   x_if_R = float(Q0.boundary_x[-1,0])
#   x_if_L = float(Q1.boundary_x[ 0,0])
#   u0_if = float(_eval_u(u0_fn, [x_if_R])[0])
#   u1_if = float(_eval_u(u1_fn, [x_if_L])[0])
#   jump  = u0_if - u1_if

#   # plot
#   fig, ax = plt.subplots(figsize=(9, 3.6))
#   # ax.axvspan(a0, b0, color="0.92", alpha=0.8, label="Ω0")
#   # ax.axvspan(a1, b1, color="0.88", alpha=0.8, label="Ω1")
#   if has_ov: ax.axvspan(a_ov, b_ov, color="gold", alpha=0.20, label="overlap")

#   ax.plot(xF, u_ex, lw=2.5, label="analytic", color="C1")
#   ax.plot(xF, u_stitch, lw=2.2, label="DD (stitched)", color="C0")
#   # ax.plot(x0g, u0, lw=1.5, label="u₀ on Ω0", color="C0", alpha=0.4)
#   # ax.plot(x1g, u1, lw=1.5, label="u₁ on Ω1", color="C0", linestyle="--", alpha=0.4)

#   ax.plot([x_if_R], [u0_if], "o", ms=6, label="u₀(b₀⁺)")
#   ax.plot([x_if_L], [u1_if], "s", ms=6, label="u₁(a₁⁻)")
#   ax.annotate(f"jump ≈ {jump:+.3e}", xy=((x_if_R+x_if_L)/2, (u0_if+u1_if)/2),
#               xytext=(10, 12), textcoords="offset points", fontsize=9)

#   ax.set_xlabel("x"); ax.set_ylabel("u(x)")
#   ax.set_title(title)
#   ax.legend(ncols=3, fontsize=9, loc="best")
#   ax.grid(True, alpha=0.25)
#   plt.tight_layout(); plt.show()

#   # simple error metric
#   l2 = np.sqrt(np.trapezoid((u_stitch - u_ex)**2, xF))
#   print(f"L2 error on plot grid ≈ {l2:.3e}")

# plot_asm_solution_with_exact(asm_out, Q0, Q1, pde)

# %%
