from hypothesis import given, settings, HealthCheck
from hypothesis.strategies import floats, integers
import numpy as np
import jax.numpy as jnp

from module.raytracing import plus_petite_racine_positive, intersection_sphere_jax, vecteurs_lumiere, intersection_sphere_vmap, calcul_lumiere_jax, calcul_lumiere_vmap
from module.setup import get_objects


def test_plus_petite_racine():
    x=plus_petite_racine_positive((12, 3, 15))
    assert x>=0

def test_intersection_sphere_jax():
    sphere1= {
            "position": jnp.array([12, -3, 6]),
            "rayon": 4,
            "texture": {
                "motif": "unicolore",
                "couleurs": [
                    "blanc",
                    "bleu"
                ]
            },
            "metallicite": 0
        }
    x=intersection_sphere_jax(jnp.array([1,0,0]), jnp.array([0,5,0]), sphere1)
    assert jnp.size(x) <=2

def test_intersection_sphere_vmap():
    liste_sphere, liste_lumiere = get_objects()
    x=intersection_sphere_vmap(jnp.array([1,0,0]), jnp.array([0,5,0]), liste_sphere)
    assert jnp.size(x) <=2

@given(floats(min_value=1, max_value=100))
def test_vecteur_lumiere(a):
    lumiere={
            "position": jnp.array([12, -3, 6]),
            "intensite": 2,
            "couleur": "jaune"
        }
    a, b, c, d=vecteurs_lumiere(jnp.array([0,0,0]), jnp.array([a,0,0]), lumiere, jnp.array([5,5,5]))
    assert 0.9<=jnp.linalg.norm(a)<=1.1 and 0.9<=jnp.linalg.norm(b)<=1.1 and 0.9<=jnp.linalg.norm(c)<=1.1 and 0.9<=jnp.linalg.norm(d)<=1.1


def test_calcul_lumiere_jax():

    lumiere={
            "position": jnp.array([12, -3, 6]),
            "intensite": 2,
            "couleur": jnp.array([0,0,1]) 
        }
    
    liste_sphere, liste_lumiere = get_objects()

    lum=calcul_lumiere_jax(jnp.array([1, 2, 3]), jnp.array([10, 20, 22]), lumiere, liste_sphere, 0)

    assert lum.all() != jnp.nan and lum.all()>=0

def test_calcul_lumiere_vmap():
    lumiere={
            "position": jnp.array([12, -3, 6]),
            "intensite": 2,
            "couleur": jnp.array([0,0,1]) 
        }
    
    liste_sphere, liste_lumiere = get_objects()

    lum=calcul_lumiere_vmap(jnp.array([1, 2, 3]), jnp.array([10, 20, 22]), liste_lumiere, liste_sphere, 0)
