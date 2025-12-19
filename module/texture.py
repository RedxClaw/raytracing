from math import fmod
import jax.numpy as jnp
from jax import lax
from jax.numpy import logical_and, logical_not, logical_or

def unicolore(theta_s, phi_s, couleurs):
    return couleurs[0]

def damier (theta_s, phi_s, couleurs):
    # L'ajout de 360 degrés est là pour contrer les problèmes de la fonction modulo lorsqu'on est autour de 0
    theta_modulo    = jnp.fmod(theta_s + 360, 20)
    phi_modulo      = jnp.fmod(phi_s + 360, 20)

    cond_theta      = theta_modulo < 10
    cond_phi        = phi_modulo < 10

    condition = logical_or(logical_and(cond_theta, logical_not(cond_phi)), logical_and(logical_not(cond_theta), cond_phi))

    return lax.cond(condition, lambda x: x[1], lambda x: x[0], couleurs)

def spirale(theta_s, phi_s, couleurs):
    # L'ajout de 360 degrés est là pour contrer les problèmes de la fonction modulo lorsqu'on est autour de 0
    angle_modulo = jnp.fmod(theta_s + phi_s + 360, 20)
    condition = jnp.logical_and(angle_modulo > 0, angle_modulo <= 10)

    return lax.cond(condition, lambda x: x[1], lambda x: x[0], couleurs)