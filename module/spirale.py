
import json
import numpy as np
from math import fmod

import jax
from jax import jit, vmap, lax
import jax.numpy as jnp
from functools import partial

@jit
def spirale(theta_s, phi_s, couleur):
    spirale_true= lambda v: jnp.array([1.0, 1.0, 1.0], float)
    spirale_false = lambda v: jnp.array([v[0], v[1], v[2]], float)

    return lax.cond(
        jnp.logical_and(0<jnp.mod(abs(theta_s+phi_s), 20) , jnp.mod(abs(theta_s+phi_s), 20)<=10),
        spirale_true, 
        spirale_false,
        couleur)
