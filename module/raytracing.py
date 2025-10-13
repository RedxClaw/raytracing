import jax
from jax import jit, vmap, lax
import jax.numpy as jnp
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import math

def intersection(F, x, ball):
    v = x - F
    v = v/norm(v)
    d = ball[0]

    a = np.dot(v, v)
    b = 2*np.dot(v, F - d)
    c = (np.dot(F - d, F - d) - ball[1]**2)

    delta = b*b - 4*a*c

    if delta >= 0:
        if (-b + math.sqrt(delta))/(2*a) > 0:
            return 1
    return 0
