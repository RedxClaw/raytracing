from hypothesis import given, settings, HealthCheck
from hypothesis.strategies import floats
import numpy as np
import jax.numpy as jnp

from module.coordonees import sphe2cart, cart2sphe, get_corners, local2global
from module.setup import get_settings



def test_coordonees_sphe2cart_1():
    v = sphe2cart(0, 0)  
    assert v[0]==1 and v[1]==0.00 and v[2]==0

def test_coordonees_spher2cart_2():
    v= sphe2cart(90,0)
    assert -0.001<=v[0]<=0.001 and 0.999<=v[1]<=1.001 and -0.001<=v[2]<=0.001


def test_coordonees_cart2sphe_1(): 
    angle = cart2sphe(jnp.array([1, 2, 3]), jnp.array([2, 3, 4]))
    assert -180.1<angle[0]<180.1 and -180.1<angle[1]<180.1


def test_local2global():
    x= local2global(300, 3, (jnp.array([0, 0, 0]), jnp.array([10, 0, 0]), jnp.array([0, -5, 0])), (400, 200))
    assert jnp.linalg.norm(x)!= jnp.nan


def test_get_corners():
    camera, resolution, taille_ecran = get_settings(720)
    u=get_corners(camera, taille_ecran)

    assert 15.9 <= jnp.linalg.norm(u[0]-u[1])<=16.1
