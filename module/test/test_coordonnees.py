

from hypothesis import given 
from hypothesis.strategies import integers, composite, floats, lists
from hypothesis.extra import numpy as nps

import numpy as np

from module import coordonees


@given(floats(min_value=-1000, max_value=1000), floats(min_value=-1000, max_value=1000))
def test_angle_to_cartesian(a, b):
    v=coordonees.angle_to_cartesian(a, b)
    assert np.dot(v, v)!=0


@given(floats(min_value=0, max_value=100), integers(min_value=0, max_value=100), integers(min_value=0, max_value=100))
def test_get_corners_distance(a, b, c):
    ul, ur, dl = coordonees.get_corners((np.array([0,0,0]), a, np.array([b, c])),np.array([16, 9]))
    n = np.linalg.norm(ul-ur)
    n1 = np.linalg.norm(ul-dl)
    assert n>=15.9 and n<=16.1 and n1>=8.9 and n1<=9.1

