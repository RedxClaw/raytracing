from hypothesis import given, settings, HealthCheck
from hypothesis.strategies import floats
import numpy as np
import jax.numpy as jnp

from module.engine import couleur_point, generation_image
from module.setup import get_objects, get_settings

def test_couleur_point():
    liste_sphere, liste_lumiere = get_objects()
    sphere={
        ""
    }
    c=couleur_point((10,8), (jnp.array([0, 0, 0]), jnp.array([10, 0, 0]), jnp.array([0, -5, 0])),(400, 200),jnp.array([4,5,6]), liste_sphere, liste_lumiere )

    assert -0.01<= c[0] <=1.1 and -0.01<= c[1] <=1.1 and -0.01<= c[2] <=1.1