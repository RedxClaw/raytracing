import matplotlib.pyplot as plt
import jax.numpy as jnp
from os import path, mkdir
from shutil import rmtree
import imageio.v2 as iio

from module import setup, engine

FPS         = 20
SECONDES    = 2
N_TOURS     = 1
N_FRAMES    = round(FPS*SECONDES/N_TOURS)

delta_degree = N_TOURS*360/N_FRAMES
theta = jnp.radians(delta_degree)
c, s = jnp.cos(theta), jnp.sin(theta)
R = jnp.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

if path.isdir('media/video'):
    rmtree('media/video')
if path.isdir('media/frames'):
    rmtree('media/frames')
mkdir('media/video')
mkdir('media/frames')

camera, resolution, taille_ecran = setup.get_settings(720)
liste_sphere, liste_lumiere = setup.get_objects()

liste_index = jnp.arange(0, N_FRAMES)

for i in range(0, N_FRAMES): 
    camera['position'] = jnp.dot(R, camera['position'])
    camera['angles'][0] += delta_degree
    ecran = engine.generation_image(camera, taille_ecran, resolution, liste_sphere, liste_lumiere)

    plt.figure()
    plt.imshow(ecran)
    plt.axis('off')
    plt.savefig(f"media/frames/frame_{i+1}.png")
    plt.close()

    print(f"Avancement : {i+1} / {N_FRAMES}", end="\r")

frames = jnp.stack([iio.imread(f"media/frames/frame_{i+1}.png") for i in range(0, N_FRAMES)], axis=0)
iio.mimwrite('media/video/animation.webp', frames, fps=FPS)

print("\nTravail terminé")