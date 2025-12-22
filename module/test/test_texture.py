from hypothesis import given, settings, HealthCheck
from hypothesis.strategies import floats
import numpy as np
import jax.numpy as jnp


from module.texture import damier, spirale


@given(floats(min_value=-1000, max_value=1000), floats(min_value=-1000, max_value=1000))
def test_damier(a,b):
    x=damier(a, b, [0.5, 0.0, 1.0])
    assert -0.0001<=x<=1.0001 

@given(floats(min_value=-1000, max_value=1000), floats(min_value=-1000, max_value=1000))
def test_spirale(a,b):
    x=spirale(a, b, [0.5, 0.0, 1.0])
    assert -0.0001<=x<=1.0001 