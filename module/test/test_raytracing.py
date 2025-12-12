
from hypothesis import given 
from hypothesis.strategies import integers, composite, floats, lists
from hypothesis.extra import numpy as nps

import numpy as np

from module import raytracing 

#TESTS MIN_POSITIF

@given(a=floats(min_value=1, max_value=1000), b=floats(min_value=-1000, max_value=1000))
def test_min_positif_est_min(a,b):
    assert( raytracing.min_positif(a,b)<=a or raytracing.min_positif(a,b)<=b)

@given(a=floats(min_value=1, max_value=1000), c=floats(min_value=-1000, max_value=-1))
def test_min_positif_est_positif(a,c):
    assert raytracing.min_positif(a,c)==a

def test_min_pos():
    assert raytracing.min_positif(1,-1)==1

def test_min_pos_1():
    assert raytracing.min_positif(0,0)==0

#TESTS VECTEURS_LUMIERE

array_strategy = nps.arrays(
    dtype="float32",
    shape=nps.array_shapes(min_dims=1, max_dims=1, min_side=3, max_side=3),
    elements=floats(min_value=-1e14, max_value=1e14, allow_nan=False, allow_infinity=False)
)

@given(array_strategy.filter(lambda n: np.linalg.norm(n) != 0),
       array_strategy.filter(lambda n: np.linalg.norm(n) != 0),
       array_strategy.filter(lambda n: np.linalg.norm(n) != 0),
       array_strategy.filter(lambda n: np.linalg.norm(n) != 0))

def test_vecteurs_lumiere_norme_autour_1(a, b, c, d):
    e,f, g, h=raytracing.vecteurs_lumiere(a, b, c, d)
    n1= np.dot(e, e)
    assert n1>=0.9 and n1<=1.1



#TESTS INTERSECTION SPHERE

camera=(np.array([0,-8,-2]), 0, 0)

lumiere1= (np.array([10,3,10]), 0.8, np.array([1,0,0]))
lumiere2= (np.array([-10, 10, 10]), 0.2, np.array([0,0,1]))

liste_lumiere= (lumiere1, lumiere2)

sphere1 = (np.array([0, 9, -2]), 1, np.array([0, 0, 1]), 500)
sphere2 = (np.array([1, 1 , 1]), 3, np.array([1, 0, 0]), 250)
sphere3 = (np.array([0, 0, 2]), 0.5, np.array([0, 1, 0]), 0)

liste_spheres = (sphere1, sphere2, sphere3)

@given(array_strategy.filter(lambda n: np.linalg.norm(n) != 0),
       array_strategy.filter(lambda n: np.linalg.norm(n) != 0))

def test_intersection_sphere_positif(a, b):
    if np.dot(a-b, a-b)==0:
        assert 1
    else:
        alpha, it=raytracing.intersection_sphere (a,b, liste_spheres, 1)
    
        assert alpha>=0 and it>=0

    