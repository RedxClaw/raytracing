import jax
from jax import jit, vmap, lax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from module import coordinates, raytracing

def solution_intersection(cam_pos, v, sphere):
    solution = []

    d = sphere[0] # Position du centre
    r = sphere[1] # Rayon de la sphère

    a = np.dot(v, v)
    b = 2 * np.dot(v, cam_pos - d)
    c = np.dot(cam_pos - d, cam_pos - d) - r*r
    delta = b*b - 4*a*c

    if delta == 0:
        solution.append(-b/(2*a))

    elif delta > 0: 
        solution.append((-b + np.sqrt(delta))/(2*a))
        solution.append((-b - np.sqrt(delta))/(2*a))

    elif delta < 0: 
        solution = None

    return solution


def couleur_pt(cam_pos, corners, liste_spheres):
    v = coordinates.local_to_global(i, j, corners, resolution)

    solutions_spheres = []

    for sphere in liste_spheres:
        if not solution_intersection(cam_pos, v, sphere[0], sphere[1], pos_pixels):
            solutions_spheres.append(solution_intersection(cam_pos,v, sphere[0], sphere[1]))
        else:
            return 0
        
    minimum = solutions_spheres.index(min(solutions_spheres))
    couleur = liste_spheres[minimum][2]

    return couleur


#caractéristiques écran:         
largeur_pixel= 256
hauteur_pixel= 144

resolution_dico = {
    '144p': (144, 256),
    '360p': (360, 640),
    '480p': (480, 854),
    '720p': (720, 1280),
    '1080': (1080, 1920)
}

resolution = resolution_dico['360p']

sphere = (np.array([0,0,0]), 1, 1) # Centre, rayon, couleur

cam = (np.array([12, 0, 0]), 180, 0) # Vecteur Position, Angles Theta & Phi (coordonnées sphériques)
sphere_1 = (np.array([1, 0, 4]), 1, 'bleu') # Vecteur Position du centre, rayon, couleur
sphere_2 = (np.array([0.2, 0, 0]), 1.5, 3)

liste_spheres = [sphere_1, sphere_2]

couleur_ecran = np.empty(resolution)

for i in range (0, resolution[0]):
    for j in range(0, resolution[1]):
        # couleur_ecran[i][j]= couleur_pt(pos_pt_focal,liste_spheres, [i,j], A, B, C, largeur_pixel, hauteur_pixel)
        pass


plt.figure()
plt.imshow(couleur_ecran, cmap='inferno')
plt.show()