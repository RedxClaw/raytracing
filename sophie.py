import jax
from jax import jit, vmap, lax
import jax.numpy as jnp
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from module import coordinates, raytracing
from module.raytracing import *


def couleur_pt(i, j, cam_pos, corners, liste_spheres):
    v = coordinates.local_to_global(i, j, corners, resolution) - cam_pos

    solutions_spheres = []
    check_sphere = False

    for sphere in liste_spheres:
        solutions_spheres.append(solution_intersection(cam_pos, v, sphere))
        
    if min(solutions_spheres) == 65536:
        return 0
    minimum = solutions_spheres.index(min(solutions_spheres))
    couleur = liste_spheres[minimum][2]

    return couleur

resolution_dico = {
    '144p': (144, 256),
    '360p': (360, 640),
    '480p': (480, 854),
    '720p': (720, 1280),
    '1080': (1080, 1920)
}

resolution = resolution_dico['360p']

screen_width = 16
screen_height = 9

cam = (np.array([12, 0, 0]), 4, (180, 0)) # Vecteur Position, Distance Focale, Angles Theta & Phi (coordonnées sphériques)
corners = coordinates.get_corners(cam, screen_height, screen_width)

sphere_1 = (np.array([0, 0, 0]), 4, 1) # Vecteur Position du centre, rayon, couleur
sphere_2 = (np.array([-3, 2, 0]), 6, 2)

liste_spheres = [sphere_1, sphere_2]

couleur_ecran = np.zeros(resolution)
couleur_liste = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]

for i in range (0, resolution[0]):
    for j in range(0, resolution[1]):
        couleur_ecran[i][j] = couleur_pt(i, j, cam[0], corners, liste_spheres)

plt.figure()
plt.imshow(couleur_ecran)
plt.show()