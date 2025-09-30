# examples/poisson1D_string_dd.py
# Domain decomposition (Robin–Schwarz ASM) for 1D Poisson with physical Robin BCs.
# %%
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable
from flax import struct  # <-- make base PDE a flax dataclass (PyTree)

from galerkinnn import FunctionState, GalerkinNN, DDQuadrature, Quadrature
from galerkinnn.formulations import DDPDE, PDE
from galerkinnn.quadratures import gauss_legendre_interval_quadrature

# ============================ Hyper-parameters ============================

seed = 42
max_bases = 6
max_epoch_basis = 50
tol_solution = 1e-7
tol_basis = 1e-7

# NN schedule (per basis index i = 1..max_bases)
N0 = 5                 # initial neurons per basis
growth = 2             # neurons growth factor per basis
A0 = 5e-3              # initial learning rate per basis
lr_decay = 1.1         # learning-rate decay per basis

# base 1D domain and physical Robin parameter
xbounds = (0.0, 1.0)
eps = 1e-3             # BC: u + eps * ∂ₙu = 0 on ∂Ω
n_train = 128          # quad points per subdomain (interior)

# ============================== PDE (string) ==============================

@struct.dataclass  # <-- this is the fix: make it a PyTree
class PoissonStringDisplacementRobinBC(PDE):
  """
  -u'' = f in (0,1)
  Physical Robin on x=0,1 (homogeneous): u + eps * ∂ₙu = 0.
  Variational form for non-vanishing test functions:
      a(u,v) = ∫ u' v' dx + (1/eps) * Σ_b u(b) v(b)
      L(v)   = ∫ f v dx
  """
  eps: float = 1e-3

  def source(self) -> Callable[[jax.Array], jax.Array]:
    # manufactured forcing (no Robin term needed in f)
    def f(X: jax.Array) -> jax.Array:
      x = X.reshape(-1, 1)
      return ((2*jnp.pi)**2 * jnp.sin(2*jnp.pi*x) +
              (4*jnp.pi)**2 * jnp.sin(4*jnp.pi*x) +
              (6*jnp.pi)**2 * jnp.sin(6*jnp.pi*x))
    return f

  def bilinear_form(self):
    eps = self.eps
    def a(u: FunctionState, v: FunctionState, quad: Quadrature) -> jax.Array:
      a_vol = jnp.einsum("nui,nvi,n->uv", u.grad_interior, v.grad_interior, quad.interior_w)
      gamma_b = (1.0 / eps) * quad.boundary_w
      mask_g = getattr(quad, "boundary_mask_global", None)
      if mask_g is not None:
        gamma_b = gamma_b * mask_g.astype(gamma_b.dtype)
      a_bnd = jnp.einsum("an,am,a->nm", u.boundary, v.boundary, gamma_b)
      return a_vol + a_bnd
    return a

  def linear_operator(self):
    f = self.source()
    def L(v: FunctionState, quad: Quadrature) -> jax.Array:
      fv = f(quad.interior_x).reshape(-1)
      return jnp.einsum("n,nv,n->v", fv, v.interior, quad.interior_w)
    return L

  def energy_norm(self):
    eps = self.eps
    def norm(v: FunctionState, quad: Quadrature) -> jax.Array:
      grad_sq = jnp.sum(v.grad_interior**2, axis=2)              # (N, n_v)
      a1 = jnp.einsum("n,ni->i", quad.interior_w, grad_sq)
      gamma_b = (1.0 / eps) * quad.boundary_w
      mask_g = getattr(quad, "boundary_mask_global", None)
      if mask_g is not None:
        gamma_b = gamma_b * mask_g.astype(gamma_b.dtype)
      a2 = jnp.einsum("n,ni->i", gamma_b, v.boundary**2)
      en2 = jnp.maximum(a1 + a2, 0.0)
      return jnp.sqrt(en2)
    return norm

# analytic solution (for plotting reference)
def u_analytic(X: jax.Array, eps_val: float):
  x = jnp.asarray(X).reshape(-1, 1)
  ur = (-24.0 * jnp.pi * eps_val * x + 12.0 * jnp.pi * eps_val) / (1.0 + 2.0 * eps_val)
  return jnp.sin(2*jnp.pi*x) + jnp.sin(4*jnp.pi*x) + jnp.sin(6*jnp.pi*x) + ur

# ========================== DD quadratures (1D) ==========================

def _owner_onehot_from_ids(owner_j_at_bndry: jax.Array,
                           neighbor_ids: Tuple[int, ...]) -> jax.Array:
  nb = owner_j_at_bndry.shape[0]
  nid = jnp.array(neighbor_ids, dtype=jnp.int32)
  return (owner_j_at_bndry.reshape(nb, 1) == nid.reshape(1, -1)).astype(jnp.float32)

def dd_overlapping_interval_quadratures(
  bounds=(0.0, 1.0), mid=0.5, overlap=0.25, ng=128
) -> Tuple[DDQuadrature, DDQuadrature]:
  a, b = map(float, bounds)
  assert b > a and 0.0 < overlap < (b - a)
  left_b  = float(jnp.clip(mid + 0.5*overlap, a, b))
  right_a = float(jnp.clip(mid - 0.5*overlap, a, b))
  assert right_a < left_b, "Invalid overlap."

  q0 = gauss_legendre_interval_quadrature((a, left_b), ng)
  q1 = gauss_legendre_interval_quadrature((right_a, b), ng)

  # Ω0: left boundary is global (−1), right owned by Ω1
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

  # Ω1: left owned by Ω0, right boundary global
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

# ========================= Tiny NN callbacks (per basis) =========================

def net_fn(X, params, activation):
  X = jnp.dot(X, params["W"]) + params["b"]
  return activation(X)

def activations_fn(i):
  s = i  # simple basis-dependent scaling
  return lambda x: jnp.tanh(s * x)

network_widths_fn = lambda i: N0 * (growth ** (i - 1))
learning_rates_fn = lambda i: A0 * (lr_decay ** (-(i - 1)))

# ================= Robin–Schwarz (impedance) ASM driver =================

TraceFn = Callable[[jax.Array], jax.Array]

def eps_from_length(Q: DDQuadrature, alpha: float = 20.0) -> float:
  """Pick eps_interface so that gamma_int ≈ alpha / H with H=subdomain length."""
  a, b = float(Q.boundary_x[0,0]), float(Q.boundary_x[-1,0])
  H = max(b - a, 1e-16)
  gamma = alpha / H
  return float(1.0 / gamma)

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

def build_pou_stitch(Q0: DDQuadrature, Q1: DDQuadrature,
                     u0_fn: TraceFn, u1_fn: TraceFn) -> TraceFn:
  a0, b0 = float(Q0.boundary_x[0,0]), float(Q0.boundary_x[-1,0])
  a1, b1 = float(Q1.boundary_x[0,0]), float(Q1.boundary_x[-1,0])
  a_ov, b_ov = max(a0, a1), min(b0, b1)
  L = max(b_ov - a_ov, 1e-16)

  def u_glob_fn(X):
    X = jnp.asarray(X).reshape(-1, 1)
    x = X[:, 0]
    in0 = (x >= a0) & (x <= b0)
    in1 = (x >= a1) & (x <= b1)
    on_ov = in0 & in1

    # PoU weights
    w0 = jnp.where(in0 & ~in1, 1.0, 0.0)
    w1 = jnp.where(in1 & ~in0, 1.0, 0.0)
    w0 = jnp.where(on_ov, (b_ov - x) / L, w0)
    w1 = jnp.where(on_ov, (x - a_ov) / L, w1)

    # Evaluate each subdomain ONLY where it has support (avoid 0*NaN)
    v0 = jnp.zeros((X.shape[0], 1))
    v1 = jnp.zeros((X.shape[0], 1))
    i0 = jnp.where(in0)[0]
    i1 = jnp.where(in1)[0]
    v0 = v0.at[i0].set(u0_fn(X[i0]))
    v1 = v1.at[i1].set(u1_fn(X[i1]))
    return w0[:, None] * v0 + w1[:, None] * v1

  return u_glob_fn

def robin_schwarz_asm_string(
    pde_base: PoissonStringDisplacementRobinBC,
    Q0: DDQuadrature, Q1: DDQuadrature,
    eps_interface: float,
    max_sweeps: int = 8,
    tol_jump: float = 5e-5,
    omega: float = 0.85,     # under-relaxation
    lr_scale: float = 0.3,   # shrink learning rates for DD solves
):
  # zero traces
  z = lambda X: jnp.zeros((X.shape[0], 1))
  zgrad = lambda X: jnp.zeros((X.shape[0], 1, Q0.dim))
  g0, g1 = z, z

  # zero initial states for local solves
  u0_state = FunctionState.from_function(z, Q0, zgrad)
  u1_state = FunctionState.from_function(z, Q1, zgrad)

  def relax(new_fn, old_fn):
    return (lambda X: (1 - omega) * old_fn(X) + omega * new_fn(X)) if omega != 1.0 else new_fn

  # learning-rate wrapper: solver calls with one arg (basis_idx); accept extras safely
  def learning_rates_fn_dd(i, *_, **__):
    return lr_scale * learning_rates_fn(i)

  # PoU weights for optional "global iterate" trace
  w0_fn, w1_fn = build_pou_weights_1d(Q0, Q1)

  history = []
  u0_fn: TraceFn = g1
  u1_fn: TraceFn = g0

  for k in range(max_sweeps):
    # Ω0 with trace from Ω1
    pde0 = DDPDE(base=pde_base, eps_interface=eps_interface, trace_fns=(g0,))
    solver0 = GalerkinNN(pde0, Q0)
    u0_state_out, *_ = solver0.solve(
      seed=seed + 1000 * k,
      u0=u0_state,
      net_fn=net_fn,
      activations_fn=activations_fn,
      network_widths_fn=network_widths_fn,
      learning_rates_fn=learning_rates_fn_dd,
      max_bases=max_bases,
      max_epoch_basis=max_epoch_basis,
      tol_solution=tol_solution,
      tol_basis=tol_basis,
    )
    # callable via simple 1D interpolation for traces/plots
    def u0_fn(X):
      xq = Q0.interior_x.reshape(-1)
      yq = u0_state_out.interior.reshape(-1)
      x = jnp.asarray(X).reshape(-1)
      return jnp.interp(x, xq, yq)[:, None]

    g1 = relax(u0_fn, g1)

    # Ω1 with trace from Ω0
    pde1 = DDPDE(base=pde_base, eps_interface=eps_interface, trace_fns=(g1,))
    solver1 = GalerkinNN(pde1, Q1)
    u1_state_out, *_ = solver1.solve(
      seed=seed + 2000 * k,
      u0=u1_state,
      net_fn=net_fn,
      activations_fn=activations_fn,
      network_widths_fn=network_widths_fn,
      learning_rates_fn=learning_rates_fn_dd,
      max_bases=max_bases,
      max_epoch_basis=max_epoch_basis,
      tol_solution=tol_solution,
      tol_basis=tol_basis,
    )
    def u1_fn(X):
      xq = Q1.interior_x.reshape(-1)
      yq = u1_state_out.interior.reshape(-1)
      x = jnp.asarray(X).reshape(-1)
      return jnp.interp(x, xq, yq)[:, None]

    g0 = relax(u1_fn, g0)

    # PoU-blended global iterate as the next trace (very stable)
    u_glob = lambda X: w0_fn(X) * u0_fn(X) + w1_fn(X) * u1_fn(X)
    g0, g1 = u_glob, u_glob

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

  return dict(Q0=Q0, Q1=Q1, u0_fn=u0_fn, u1_fn=u1_fn, logs=history)

# ================================ main =================================

# DD quadratures on [0,1]
Q0, Q1 = dd_overlapping_interval_quadratures(bounds=xbounds, mid=0.5, overlap=0.25, ng=n_train)

# base PDE with physical Robin on ∂Ω  (now a dataclass, so JAX-friendly)
pde_base = PoissonStringDisplacementRobinBC(eps=eps)

# interface penalty (gamma ≈ α/H)
def eps_from_length(Q: DDQuadrature, alpha: float = 20.0) -> float:
  a, b = float(Q.boundary_x[0,0]), float(Q.boundary_x[-1,0])
  H = max(b - a, 1e-16)
  gamma = alpha / H
  return float(1.0 / gamma)

eps0 = eps_from_length(Q0, alpha=20.0)
eps1 = eps_from_length(Q1, alpha=20.0)
eps_interface = 0.5 * (eps0 + eps1)

asm = robin_schwarz_asm_string(
  pde_base=pde_base,
  Q0=Q0, Q1=Q1,
  eps_interface=eps_interface,
  max_sweeps=8,
  tol_jump=5e-5,
  omega=0.85,
  lr_scale=0.3,
)

# stitched global curve for plotting
u_glob_fn = build_pou_stitch(Q0, Q1, asm["u0_fn"], asm["u1_fn"])

# dense grid
xF = np.linspace(xbounds[0], xbounds[1], 800)
u_num = np.array(jnp.squeeze(u_glob_fn(xF[:, None])))
u_exa = np.array(jnp.squeeze(u_analytic(jnp.array(xF)[:, None], eps)))

# plot numerical (stitched) vs analytic
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot(xF, u_exa, lw=2.5, label="analytic", color="C1")
ax.plot(xF, u_num, lw=2.2, label="DD (stitched PoU)", color="C0")

# shade subdomains & overlap
a0, b0 = float(Q0.boundary_x[0,0]), float(Q0.boundary_x[-1,0])
a1, b1 = float(Q1.boundary_x[0,0]), float(Q1.boundary_x[-1,0])
a_ov, b_ov = max(a0, a1), min(b0, b1)
ax.axvspan(a0, b0, color="0.92", alpha=0.8, label="Ω0")
ax.axvspan(a1, b1, color="0.88", alpha=0.8, label="Ω1")
ax.axvspan(a_ov, b_ov, color="gold", alpha=0.20, label="overlap")

# interface markers
x_if_R = float(Q0.boundary_x[-1,0])
x_if_L = float(Q1.boundary_x[ 0,0])
u0_if = float(jnp.squeeze(asm["u0_fn"](jnp.array([[x_if_R]]))))
u1_if = float(jnp.squeeze(asm["u1_fn"](jnp.array([[x_if_L]]))))
ax.plot([x_if_R], [u0_if], "o", ms=6, label="u₀(b₀⁺)")
ax.plot([x_if_L], [u1_if], "s", ms=6, label="u₁(a₁⁻)")
ax.annotate(f"jump ≈ {u0_if - u1_if:+.3e}",
            xy=((x_if_R+x_if_L)/2, (u0_if+u1_if)/2),
            xytext=(10, 12), textcoords="offset points", fontsize=9)

ax.set_xlabel("x")
ax.set_ylabel("u(x)")
ax.set_title("Poisson 1D (physical Robin) — Robin–Schwarz ASM vs analytic")
ax.legend(ncols=3, fontsize=9, loc="best")
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.show()

# interface jump history (log scale)
hist = asm["logs"]
k = np.array([h["sweep"] for h in hist])
j = np.array([h["jump"]  for h in hist])
fig, ax = plt.subplots(figsize=(6.0, 3.0))
ax.plot(k, np.abs(j), marker="o")
ax.set_yscale("log")
ax.set_xlabel("sweep")
ax.set_ylabel("|jump|")
ax.set_title("ASM interface jump per sweep")
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.show()

# %%