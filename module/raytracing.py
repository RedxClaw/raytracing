import jax
from jax import jit, vmap, lax
import jax.numpy as jnp
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

def intersection(F, x, ball):
    v = x - F
    v = v/norm(v)

    if (4*np.dot(v, F))**2 - 4*np.dot(v, v)*(np.dot(F, F) - ball[1]) >= 0:
        return 1
    return 0
