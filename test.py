import jax
from jax import vmap
import jax.numpy as jnp

def test_fun(c):
    c = c.at[0].set(1)
    c = c.at[2].set(4)
    c = c.at[3].set(2)

    return c

for i in range(2):
    print(i)

matrix = jnp.zeros([5, 3, 3])

test_row = vmap(test_fun, in_axes=1)
test_matrix = vmap(test_row, in_axes=2)

print(test_matrix(matrix))
