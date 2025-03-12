import numpy as np
import jax.numpy as jnp

def gauss_lengendre_quad(a, b, ng):
    # a, b -- 1D domain
    # ng -- number of nodes for training
    x, w = np.polynomial.legendre.leggauss(ng)
    x = 0.5 * (b - a) * x + 0.5 * (b + a)  # Translation from [-1, 1] to [a, b]
    w = 0.5 * (b - a) * w  # Scale quadrature weights
    return jnp.array(x).reshape(-1, 1), jnp.array(w).reshape(-1, 1)