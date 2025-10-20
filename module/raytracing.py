import jax
from jax import jit, vmap, lax
import jax.numpy as jnp
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import sys

def plus_petit_positif(liste):
    temp = 65536
    for element in liste:
        if temp > element and element > 0:
            temp = element

    return temp

def solution_intersection(cam_pos, v, sphere):
    d = sphere[0] # Position du centre
    r = sphere[1] # Rayon de la sphère

    a = np.dot(v, v)
    b = 2 * np.dot(v, cam_pos - d)
    c = np.dot(cam_pos - d, cam_pos - d) - r*r

    delta = b*b - 4*a*c

    solution = 65536

    if delta == 0:
        s = -b/(2*a)
        if s > 0 : solution = s
    elif delta > 0: 
        solution = plus_petit_positif([(-b - np.sqrt(delta))/(2*a), (-b + np.sqrt(delta))/(2*a)])

    return solution
