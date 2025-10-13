import jax
from jax import jit, vmap, lax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from module import coordinates, raytracing

focal_length = 8

screen_width = 8
screen_height = 4.5

resolution_dico = {
    '144p': (144, 256),
    '360p': (360, 640),
    '480p': (480, 854),
    '720p': (720, 1280),
    '1080': (1080, 1920)
}

resolution = resolution_dico['360p']

cam = (np.array([12, 0, 0]), np.array([-1, 0, 0]))
ball = (np.array([0, 0, 0]), 1, 'blue')

S = np.zeros(resolution)
corners = coordinates.get_corners(cam, focal_length, screen_height, screen_width)

for i in range(0, resolution[1]):
    for j in range(0, resolution[0]):
        x = coordinates.local_to_global(i, j, corners, resolution)
        S[i][j] = raytracing.intersection(cam[0], x, ball)

plt.figure()
plt.imshow(S, cmap='hot', origin='lower')
plt.show()










