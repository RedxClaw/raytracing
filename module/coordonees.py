import jax.numpy as jnp
from jax.numpy import acos, cos, sin, radians, degrees

"""
    sphe2cart(theta, phi)

Passe des coordonnées sphériques aux coordonnées cartésiennes

# INPUT :
* `theta`   L'angle du plan OXY
* `phi`     L'angle vertical

# OUTPUT :
* `vec`     Vecteur de norme 1
"""
def sphe2cart(theta, phi):
    theta = radians(theta)
    phi = radians(phi)

    c_phi = cos(phi)
    vec = jnp.array([c_phi*cos(theta), c_phi*sin(theta), sin(phi)])
    return vec

"""
    cart2sphe(point, sphere_centre)

Calcul les coordonées sphériques d'un point par rapport au centre d'une sphère donnée

# INPUT :
* `point`           Vecteur de dimension 3
* `sphere_centre`   Vecteur position de dimension 3 qui pointe vers le centre de la sphère

# OUTPUT :
* Une paire d'angles (`theta`, `phi`)
"""
def cart2sphe(point, sphere_center):
    vec = point - sphere_center

    x = vec[0]
    y = vec[1]
    z = vec[2]

    theta_s = jnp.rad2deg(jnp.acos(z/jnp.linalg.norm(vec)))
    phi_s = jnp.rad2deg(jnp.sign(y)*jnp.acos(x/jnp.linalg.norm(jnp.array([x, y]))))

    return jnp.array([theta_s, phi_s])

"""
    get_corners(camera, taille_ecran)

Calcul 3 des 4 coins de l'écran simulé en partant du point focal de la caméra

# INPUT :
* `camera`          Dictionnaire contenant :
    * `'position'`          le vecteur position du point focal
    * `'distance_focale'`   la distance entre le point focal et l'écran simulé
    * `'angles'`            paire d'angles permettant d'obtenir l'orientation de la caméra
* `taille_ecran`    Dictionnaire de réels qui donne les dimensions de l'écran simulé

# OUTPUT :
* 3 vecteurs normalisés de dimension 3 :
    * `ul`          Vecteur du haut, à gauche (Up-Left)
    * `ur`          Vecteur du haut, à droite (Up-Right)
    * `dl`          Vecteur du bas, à gauche (Down-Left)
"""
def get_corners(camera, taille_ecran):
    largeur = taille_ecran['largeur']
    hauteur = taille_ecran['hauteur']

    cam_pos = camera["position"]
    focal = camera["distance_focale"]
    theta, psi = camera["angles"]

    vec_1 = sphe2cart(theta, psi)
    vec_2 = jnp.array([-vec_1[1], vec_1[0], 0])
    vec_3 = jnp.cross(vec_1, vec_2)

    vec_1 = vec_1/jnp.linalg.norm(vec_1)
    vec_2 = vec_2/jnp.linalg.norm(vec_2)
    vec_3 = vec_3/jnp.linalg.norm(vec_3)

    ul = cam_pos + vec_1*focal - largeur*vec_2/2 + hauteur*vec_3/2
    ur = cam_pos + vec_1*focal + largeur*vec_2/2 + hauteur*vec_3/2
    dl = cam_pos + vec_1*focal - largeur*vec_2/2 - hauteur*vec_3/2

    return (ul, ur, dl)

"""
    local2global(i, j, corners, resolution)

Passage des coordonnées locales du repère barycentrique autour de la caméra aux coorfonnées globales de l'espace simulé
    en interpolant linéairement l'espace grâce aux 3 coins de l'écran simulé et la résolution de l'écran

# INPUT :
* `i, j`        Coordonnées locales d'un point (i -> horizontale ; j -> verticale)
* `corners`     Les 3 coins de l'écran simulé utilisés pour l'interpolation linéaire
* `resolution`  La résolution de l'écran simulé (ce paramètre influe directement sur le nombre de points calculés)

# OUTPUT :
* `x`           Vecteur de dimension 3 dans l'espace simulé
"""
def local2global(i, j, corners, resolution):
    t1 = i/resolution[0]
    t2 = j/resolution[1]

    x = (1 - t1 - t2) * corners[0] + t1*corners[2] + t2*corners[1]

    return x