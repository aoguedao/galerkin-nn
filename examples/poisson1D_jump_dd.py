# examples/poisson1d_jump_DD.py
# Domain decomposition (Robin–Schwarz ASM) for 1D Poisson with piecewise k and physical Robin BCs.
# %%
import jax
import jax.numpy as jnp
from flax import struct
from typing import Callable, Tuple
import numpy as np
import matplotlib.pyplot as plt

from galerkinnn import FunctionState, PDE, Quadrature, GalerkinNN, DDQuadrature
from galerkinnn.quadratures import gauss_legendre_interval_quadrature
from galerkinnn.formulations import DDPDE
from galerkinnn.utils import make_u_fn

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

# =============== DD quadratures (two overlapping intervals) ===================

def _owner_onehot_from_ids(owner_j_at_bndry: jax.Array,
                           neighbor_ids: Tuple[int, ...]) -> jax.Array:
  nb = owner_j_at_bndry.shape[0]
  nid = jnp.array(neighbor_ids, dtype=jnp.int32)  # (Jn,)
  return (owner_j_at_bndry.reshape(nb, 1) == nid.reshape(1, -1)).astype(jnp.float32)

def dd_overlapping_interval_quadratures(
    bounds=(0.0, 1.0), mid=0.5, overlap=0.25, ng=128
) -> Tuple[DDQuadrature, DDQuadrature]:
  a, b = map(float, bounds)
  assert b > a and 0.0 < overlap < (b - a)
  left_b  = float(jnp.clip(mid + 0.5 * overlap, a, b))
  right_a = float(jnp.clip(mid - 0.5 * overlap, a, b))
  assert right_a < left_b, "Invalid overlap: subdomains do not overlap."

  q0 = gauss_legendre_interval_quadrature((a, left_b), ng)
  q1 = gauss_legendre_interval_quadrature((right_a, b), ng)

  # Ω0 metadata: left boundary is global, right boundary owned by Ω1
  sub0_owner = jnp.array([-1, 1], dtype=jnp.int32)
  sub0_neigh = (1,)
  Q0 = DDQuadrature(
    dim=q0.dim, shape=q0.shape,
    interior_x=q0.interior_x, interior_w=q0.interior_w,
    boundary_x=q0.boundary_x, boundary_w=q0.boundary_w,
    boundary_tangent=q0.boundary_tangent, boundary_normal=q0.boundary_normal,
    subdomain_id=0, neighbor_ids=sub0_neigh,
    owner_j_at_bndry=sub0_owner,
    boundary_owner_onehot=_owner_onehot_from_ids(sub0_owner, sub0_neigh),
    boundary_mask_global=(sub0_owner == -1),
  )

  # Ω1 metadata: left boundary owned by Ω0, right boundary is global
  sub1_owner = jnp.array([0, -1], dtype=jnp.int32)
  sub1_neigh = (0,)
  Q1 = DDQuadrature(
    dim=q1.dim, shape=q1.shape,
    interior_x=q1.interior_x, interior_w=q1.interior_w,
    boundary_x=q1.boundary_x, boundary_w=q1.boundary_w,
    boundary_tangent=q1.boundary_tangent, boundary_normal=q1.boundary_normal,
    subdomain_id=1, neighbor_ids=sub1_neigh,
    owner_j_at_bndry=sub1_owner,
    boundary_owner_onehot=_owner_onehot_from_ids(sub1_owner, sub1_neigh),
    boundary_mask_global=(sub1_owner == -1),
  )
  return Q0, Q1

# ====================== PDE: piecewise k, physical Robin ======================

@struct.dataclass
class Poisson1DJumpRobin(PDE):
  """
  -(k u')' = f in (0,1), homogeneous Robin via 1/eps on the physical boundary:
      a(u,v) = ∫ k u' v' dx + (1/eps) * sum_{b ∈ ∂Ω} u(b) v(b)
      L(v)   = ∫ f v dx
  k is piecewise-constant with a jump at x0. Exact u is piecewise quadratic
  with continuity of u and flux k u' at x0, and Robin on x=0,1.
  """
  k1: float = 1.0
  k2: float = 10.0
  x0: float = 0.5
  eps: float = 1e-3
  a_curv: float = 1.0  # common quadratic coefficient 'a' (u''=2a)

  def k(self, X: jax.Array) -> jax.Array:
    x = X.reshape(-1)
    return jnp.where(x <= self.x0, self.k1, self.k2)[:, None]  # (N,1)

  def source(self) -> Callable[[jax.Array], jax.Array]:
    """
    Choose f so that u'' = 2*a_curv in each piece. Since k is constant per piece,
    f = -(k u'') = -2*a_curv * k(x).
    """
    a = self.a_curv
    def f(X: jax.Array) -> jax.Array:
      kx = self.k(X)  # (N,1)
      return -2.0 * a * kx
    return f

  def bilinear_form(self):
    eps = self.eps
    def a(u: FunctionState, v: FunctionState, quad: Quadrature) -> jax.Array:
      # interior: ∫ k u' v' dx
      kx = self.k(quad.interior_x).reshape(-1)  # (N,)
      a_vol = jnp.einsum("nui,nvi,n->uv", u.grad_interior, v.grad_interior,
                         quad.interior_w * kx)
      # physical boundary: (1/eps) * Σ_b u(b) v(b)
      gamma_b = (1.0 / eps) * quad.boundary_w              # (Nb,)
      mask_g  = getattr(quad, "boundary_mask_global", None)
      if mask_g is not None:
        gamma_b = gamma_b * mask_g.astype(gamma_b.dtype)
      a_bnd = jnp.einsum("an,am,a->nm", u.boundary, v.boundary, gamma_b)
      return a_vol + a_bnd
    return a

  def linear_operator(self):
    f = self.source()
    def L(v: FunctionState, quad: Quadrature) -> jnp.ndarray:
      fvals = f(quad.interior_x).reshape(-1)                 # (N,)
      return jnp.einsum("n,nv,n->v", fvals, v.interior, quad.interior_w)
    return L

  def energy_norm(self):
    eps = self.eps
    def norm(v: FunctionState, quad: Quadrature) -> jax.Array:
      kx = self.k(quad.interior_x).reshape(-1)               # (N,)
      grad_sq = jnp.sum(v.grad_interior**2, axis=2)          # (N, n_v)
      a1 = jnp.einsum("n,ni->i", quad.interior_w * kx, grad_sq)
      b_sq = v.boundary**2                                   # (Nb, n_v)
      gamma_b = (1.0 / eps) * quad.boundary_w
      mask_g  = getattr(quad, "boundary_mask_global", None)
      if mask_g is not None:
        gamma_b = gamma_b * mask_g.astype(gamma_b.dtype)
      a2 = jnp.einsum("n,ni->i", gamma_b, b_sq)
      en2 = jnp.maximum(a1 + a2, jnp.array(0., a1.dtype))
      return jnp.sqrt(en2)
    return norm

# ---------------- Analytic solution for comparison ----------------

def exact_coeffs(k1: float, k2: float, x0: float, eps: float, a: float) -> Tuple[float,float,float,float]:
  """
  u_-(x)=a x^2 + bL x + cL on [0,x0], u_+(x)=a x^2 + bR x + cR on [x0,1].
  Conditions:
    (i) u - eps*k1*u'(0) = 0 at x=0   (physical Robin with outward normal -1)
    (ii) u + eps*k2*u'(1) = 0 at x=1  (outward normal +1)
    (iii) continuity: u_-(x0) = u_+(x0)
    (iv) flux cont.:  k1 u_-'(x0) = k2 u_+'(x0)
  Returns (bL, cL, bR, cR).
  """
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

# ================== Tiny NN “blocks” expected by GalerkinNN ===================

def net_fn(X, params, activation):
  X = jnp.dot(X, params["W"]) + params["b"]
  return activation(X)

def activations_fn(i):
  s = i
  return lambda x: jnp.tanh(s * x)

network_widths_fn = lambda i: N * (r ** (i - 1))
learning_rates_fn = lambda i: A * (rho ** (-(i - 1)))

# ================== Robin–Schwarz (impedance) ASM driver =====================

def robin_schwarz_asm(
  base_pde,
  Q0, Q1,
  eps_interface=1e-5,
  max_sweeps=6,
  tol_jump=5e-4,
  omega=1.0,
  seed0=42, seed1=43,
):
  # zero initial traces / states
  z = lambda X: jnp.zeros((X.shape[0], 1))
  gradz = lambda X: jnp.zeros((X.shape[0], 1, Q0.dim))
  u0_state = FunctionState.from_function(z, Q0, gradz)
  u1_state = FunctionState.from_function(z, Q1, gradz)
  g0 = z; g1 = z

  # PoU weights for trace blending
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

  w0_fn, w1_fn = build_pou_weights_1d(Q0, Q1)

  def relax(new_fn, old_fn):
    return (lambda X: (1 - omega) * old_fn(X) + omega * new_fn(X)) if omega != 1.0 else new_fn

  history = []
  u0_fn = g1
  u1_fn = g0

  for k in range(max_sweeps):
    # Ω0
    pde0 = DDPDE(base=base_pde, eps_interface=eps_interface, trace_fns=(g0,))
    solver0 = GalerkinNN(pde0, Q0)
    u0_state_out, u0_coeff, *_0, basis_coeff_list0, sigma_net_fn_list0 = solver0.solve(
      seed=seed0 + 100*k,
      u0=u0_state,
      net_fn=net_fn,
      activations_fn=activations_fn,
      network_widths_fn=network_widths_fn,
      learning_rates_fn=learning_rates_fn,
      max_bases=max_bases,
      max_epoch_basis=max_epoch_basis,
      tol_solution=tol_solution,
      tol_basis=tol_basis,
    )
    u0_fn = make_u_fn(sigma_net_fn_list0, u_coeff=u0_coeff, basis_coeff_list=basis_coeff_list0)
    g1 = relax(u0_fn, g1)

    # Ω1
    pde1 = DDPDE(base=base_pde, eps_interface=eps_interface, trace_fns=(g1,))
    solver1 = GalerkinNN(pde1, Q1)
    u1_state_out, u1_coeff, *_1, basis_coeff_list1, sigma_net_fn_list1 = solver1.solve(
      seed=seed1 + 100*k,
      u0=u1_state,
      net_fn=net_fn,
      activations_fn=activations_fn,
      network_widths_fn=network_widths_fn,
      learning_rates_fn=learning_rates_fn,
      max_bases=max_bases,
      max_epoch_basis=max_epoch_basis,
      tol_solution=tol_solution,
      tol_basis=tol_basis,
    )
    u1_fn = make_u_fn(sigma_net_fn_list1, u_coeff=u1_coeff, basis_coeff_list=basis_coeff_list1)

    # PoU-blended global iterate for the next traces
    u_glob = lambda X: w0_fn(X) * u0_fn(X) + w1_fn(X) * u1_fn(X)
    g0 = u_glob
    g1 = u_glob

    # interface diagnostic
    x_if_R = Q0.boundary_x[-1:]   # (1,1)
    x_if_L = Q1.boundary_x[:1]    # (1,1)
    u0_if = jnp.squeeze(u0_fn(x_if_R)).item()
    u1_if = jnp.squeeze(u1_fn(x_if_L)).item()
    jump  = u0_if - u1_if
    print(f"[sweep {k+1:02d}] u0(b0^+)={u0_if:+.6e}, u1(a1^-)={u1_if:+.6e}, jump={jump:+.6e}")
    history.append(dict(sweep=k+1, u0_if=u0_if, u1_if=u1_if, jump=jump))
    if abs(jump) < tol_jump:
      break

  return dict(u0_fn=u0_fn, u1_fn=u1_fn, logs=history)

# =========================== Run ASM on two subs ==============================

Q0, Q1 = dd_overlapping_interval_quadratures(bounds=bounds, mid=mid, overlap=overlap, ng=n_train)
pde = Poisson1DJumpRobin(k1=k1, k2=k2, x0=x0, eps=eps, a_curv=a_curv)

asm_out = robin_schwarz_asm(
  base_pde=pde, Q0=Q0, Q1=Q1,
  eps_interface=1e-5,   # try 1e-4..1e-2 to tighten/loosen interface Dirichlet
  max_sweeps=6,
  tol_jump=5e-4,
  omega=1.0,
  seed0=42, seed1=43,
)

# ========================= Plot stitched vs analytic ==========================

def _eval_u(u_fn, x1d):
  X = jnp.array(x1d, dtype=jnp.float32).reshape(-1, 1)
  val = jnp.asarray(u_fn(X))
  if val.ndim == 2 and val.shape[1] == 1:
    val = val[:, 0]
  return np.array(val)

def plot_asm_solution_with_exact(asm_out, Q0, Q1, pde, n_plot=800,
                                 title="Poisson 1D (physical Robin) — Robin–Schwarz ASM vs analytic"):
  u0_fn = asm_out["u0_fn"]; u1_fn = asm_out["u1_fn"]

  # subdomain bounds
  a0, b0 = float(Q0.boundary_x[0,0]), float(Q0.boundary_x[-1,0])
  a1, b1 = float(Q1.boundary_x[0,0]), float(Q1.boundary_x[-1,0])

  # overlap
  a_ov, b_ov = max(a0, a1), min(b0, b1)
  has_ov = b_ov > a_ov + 1e-14

  # dense grids
  x0g = np.linspace(a0, b0, max(200, n_plot//2))
  x1g = np.linspace(a1, b1, max(200, n_plot//2))
  xF = np.linspace(min(a0, a1), max(b0, b1), n_plot)

  u0 = _eval_u(u0_fn, x0g)
  u1 = _eval_u(u1_fn, x1g)

  # stitched curve (avoid NaNs by only sampling where defined)
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

  # analytic solution
  u_exact = u_exact_fn(k1=pde.k1, k2=pde.k2, x0=pde.x0, eps=pde.eps, a=pde.a_curv)
  u_ex = _eval_u(u_exact, xF)

  # interface values for annotation
  x_if_R = float(Q0.boundary_x[-1,0])
  x_if_L = float(Q1.boundary_x[ 0,0])
  u0_if = float(_eval_u(u0_fn, [x_if_R])[0])
  u1_if = float(_eval_u(u1_fn, [x_if_L])[0])
  jump  = u0_if - u1_if

  # plot
  fig, ax = plt.subplots(figsize=(9, 3.6))
  ax.axvspan(a0, b0, color="0.92", alpha=0.8, label="Ω0")
  ax.axvspan(a1, b1, color="0.88", alpha=0.8, label="Ω1")
  if has_ov: ax.axvspan(a_ov, b_ov, color="gold", alpha=0.20, label="overlap")

  ax.plot(xF, u_ex, lw=2.5, label="analytic", color="C1")
  ax.plot(xF, u_stitch, lw=2.2, label="DD (stitched)", color="C0")
  ax.plot(x0g, u0, lw=1.5, label="u₀ on Ω0", color="C0", alpha=0.4)
  ax.plot(x1g, u1, lw=1.5, label="u₁ on Ω1", color="C0", linestyle="--", alpha=0.4)

  ax.plot([x_if_R], [u0_if], "o", ms=6, label="u₀(b₀⁺)")
  ax.plot([x_if_L], [u1_if], "s", ms=6, label="u₁(a₁⁻)")
  ax.annotate(f"jump ≈ {jump:+.3e}", xy=((x_if_R+x_if_L)/2, (u0_if+u1_if)/2),
              xytext=(10, 12), textcoords="offset points", fontsize=9)

  ax.set_xlabel("x"); ax.set_ylabel("u(x)")
  ax.set_title(title)
  ax.legend(ncols=3, fontsize=9, loc="best")
  ax.grid(True, alpha=0.25)
  plt.tight_layout(); plt.show()

  # simple error metric
  l2 = np.sqrt(np.trapezoid((u_stitch - u_ex)**2, xF))
  print(f"L2 error on plot grid ≈ {l2:.3e}")

plot_asm_solution_with_exact(asm_out, Q0, Q1, pde)
