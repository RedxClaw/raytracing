import matplotlib.pyplot as plt
from jax import vmap, jit, lax, debug
import jax.numpy as jnp

from module import coordonees, raytracing, setup

"""Renvoie la couleur d'un pixel donné de l'écran simulé"""
def couleur_point(ij, corners, resolution, cam_pos, liste_sphere, liste_lumiere):
    i, j = ij
    x = coordonees.local2global(i, j, corners, resolution)
    v = x - cam_pos

    liste_alpha = raytracing.intersection_sphere_vmap(cam_pos, v, liste_sphere)

    iteration = jnp.argsort(liste_alpha)[0]
    alpha = liste_alpha[iteration]

    p = cam_pos + alpha*v
    theta_s, phi_s = coordonees.cart2sphe(p, liste_sphere['position'][iteration])

    couleur_sphere = lax.switch(liste_sphere['motif'][iteration], setup.liste_ft_textures, theta_s, phi_s, liste_sphere['couleurs'][iteration])

    true_fun = lambda iteration: couleur_sphere*raytracing.calcul_lumiere_vmap(cam_pos, p, liste_lumiere, liste_sphere, iteration)
    false_fun = lambda iteration: jnp.array([0., 0., 0.])
    
    return lax.cond(jnp.logical_not(jnp.isnan(alpha)), true_fun, false_fun, iteration)

"""
    generation_image(camera, taille_ecran, resolution, liste_sphere, liste_lumiere)

# INPUT :
* `camera`          Dictionnaire contenant :
    * `'position'`          Le vecteur position du point focal
    * `'distance_focale'`   La distance entre le point focal et l'écran simulé
    * `'angles'`            Paire d'angles permettant d'obtenir l'orientation de la caméra
* `taille_ecran`    Paire de réels qui donne la taille de l'écran simulé
* `resolution`      Paire d'entiers positifs donnant la résolution de l'écran simulé
* `liste_sphere`    Liste des sphères présentes dans la scène
* `liste_lumiere`   Liste des lumières présentes dans la scène

# OUTPUT :
* `resultat`        Matrice de la taille de l'écran contenant les triplets RGB de chaque pixel de l'image
"""
def generation_image(camera, taille_ecran, resolution, liste_sphere, liste_lumiere):
    ecran = setup.setup_ecran(resolution)
    corners = coordonees.get_corners(camera, taille_ecran)
    cam_pos = camera["position"]

    fun = lambda x: couleur_point(x, corners, resolution, cam_pos, liste_sphere, liste_lumiere)

    fun_row = vmap(fun, in_axes=1)
    fun_jax = vmap(fun_row, in_axes=2)

    resultat = fun_jax(ecran)

    return jnp.where(resultat > 0, resultat, 0)
