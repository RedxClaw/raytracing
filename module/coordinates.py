import jax
from jax import jit, vmap, lax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

def get_corners(cam, focal_length, screen_height, screen_width):
    vec1 = cam[1]
    vec2 = np.array([vec1[1], vec1[0], 0])
    vec3 = np.cross(vec1, vec2)

    vec1 = vec1/np.linalg.norm(vec1)
    vec2 = vec2/np.linalg.norm(vec2)
    vec3 = vec3/np.linalg.norm(vec3)

    ul = cam[0] + focal_length*vec1 - (screen_width/2)*vec2 + (screen_height/2)*vec3
    ur = cam[0] + focal_length*vec1 + (screen_width/2)*vec2 + (screen_height/2)*vec3
    dl = cam[0] + focal_length*vec1 - (screen_width/2)*vec2 - (screen_height/2)*vec3

    corners = (ul, ur, dl)
    return corners

def local_to_global(i, j, corners, resolution):
    t1 = i/resolution[0]
    t2 = j/resolution[1]

    x = corners[0]*(1 - t1 - t2) + t1*corners[2] + t2*corners[1]

    return x