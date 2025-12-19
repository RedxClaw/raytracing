import jax.numpy as jnp
from jax import lax, vmap
import numpy as np

def plus_petite_racine_positive(x):
    a, b, delta = x

    solutions = jnp.array([(-b + jnp.sqrt(delta))/(2*a), (-b - jnp.sqrt(delta))/(2*a)], dtype='float32')

    solutions = jnp.where(solutions > 0, solutions, jnp.nan)
    return jnp.sort(solutions)[0]

def intersection_sphere_jax(x, v, sphere):
    d = sphere["position"]
    r = sphere["rayon"]

    a = jnp.dot(v, v)
    b = 2*jnp.dot(v, x-d)
    c = jnp.dot(x-d,x-d) - r*r

    delta = b*b - 4*a*c
    
    return jnp.select(condlist=[delta > 0, delta == 0],
                      choicelist=[plus_petite_racine_positive((a, b, delta)), -b/(2*a)],
                      default=jnp.nan)

def intersection_sphere_vmap(x, v, liste_sphere):
    f = vmap(intersection_sphere_jax, in_axes=(None, None, 0))
    resultat = jnp.array(f(x, v, liste_sphere), dtype='float32')
    return jnp.where(resultat > 0, resultat, jnp.nan)

"""Calculs des 4 vecteurs nécessaires aux calculs du Phong-Shading"""
def vecteurs_lumiere(cam_pos, p, lumiere, sphere_pos):
    n = p - sphere_pos
    vl = lumiere["position"] - p
    vc = cam_pos - p
    vr = 2*jnp.dot(vl, n)*n - vl

    n = n/jnp.linalg.norm(n)
    vl = vl/jnp.linalg.norm(vl)
    vc = vc/jnp.linalg.norm(vc)
    vr = vr/jnp.linalg.norm(vr)

    return n, vl, vc, vr

"""Calcul la contribution d'une lumière pour un seul pixel de l'écran"""
def calcul_lumiere_jax(cam_pos, p, lumiere, liste_sphere, indice_sphere):
    k_a = 0.4
    k_d = 0.5
    k_s = 0.6

    lumiere_pos = lumiere["position"]
    beta = liste_sphere["metallicite"][indice_sphere]

    intensite = 40*lumiere["intensite"]/jnp.dot(p - lumiere_pos, p - lumiere_pos)
    factor = k_a * intensite

    liste_alpha = intersection_sphere_vmap(p, lumiere_pos - p, liste_sphere)
    liste_alpha = liste_alpha.at[indice_sphere].set(jnp.nan)
    alpha = jnp.sort(liste_alpha)[0]

    n, vl, vc, vr = vecteurs_lumiere(cam_pos, p, lumiere, liste_sphere["position"][indice_sphere])
    quantite_lumiere_point = lax.cond(
        pred        = jnp.logical_or(jnp.isnan(alpha), alpha >= 1), 
        true_fun    = lambda x: x + k_d * jnp.dot(vl, n) * intensite + k_s * (jnp.dot(vr, vc)**beta) * intensite,
        false_fun   = lambda x: x,
        operand     = factor)
    
    return quantite_lumiere_point*lumiere['couleur']

"""
    calcul_lumiere_vmap(cam_pos, p, liste_lumiere, liste_sphere, indice_sphere)

# INPUT :
* `cam_pos`         Position du point focal
* `p`               Position du point d'intersection sur la sphère
* `liste_sphere`    Liste des sphères présentes dans la scène
* `liste_lumiere`   Liste des lumières présentes dans la scène
* `indice_sphere`   Sphere sur laquelle le point p se situe

# OUTPUT :
* `lumiere`         Somme de l'ensemble des contributions des lumières pour la couleur et la luminosité du point p
"""
def calcul_lumiere_vmap(cam_pos, p, liste_lumiere, liste_sphere, indice_sphere):
    f = vmap(calcul_lumiere_jax, in_axes=(None, None, 0, None, None))
    liste_intensite = f(cam_pos, p, liste_lumiere, liste_sphere, indice_sphere)
    lumiere = jnp.sum(liste_intensite, axis=0)
    return jnp.where(lumiere < 1, lumiere, 1)
