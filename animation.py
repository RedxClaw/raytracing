import matplotlib.pyplot as plt
import jax.numpy as jnp
from os import path, mkdir
from shutil import rmtree
import imageio.v2 as iio

from module import setup, engine

# Paramètres généraux
FICHIER_PARAMETRES          = "./module/json/settings.json"
FICHIER_OBJETS              = "./module/json/objets_1.json"

HAUTEUR_RESOLUTION          = 1080
N_ECHANTILLON_STOCHASTIQUE  = 1

# Paramètres spécifiques à l'animation
FPS         = 20
SECONDES    = 5
N_TOURS     = 1
N_FRAMES    = round(FPS*SECONDES/N_TOURS)
FORMAT      = ".webp"

if not path.isdir('media'):
    mkdir('media')
if path.isdir('media/video'):
    rmtree('media/video')
if path.isdir('media/frames'):
    rmtree('media/frames')
mkdir('media/video')
mkdir('media/frames')

delta_degree = N_TOURS*360/N_FRAMES
theta = jnp.radians(delta_degree)
c, s = jnp.cos(theta), jnp.sin(theta)
R = jnp.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

camera, resolution, taille_ecran, coefficients_lumiere = setup.get_settings(FICHIER_PARAMETRES, HAUTEUR_RESOLUTION)
liste_sphere, liste_lumiere = setup.get_objects(FICHIER_OBJETS)

liste_index = jnp.arange(0, N_FRAMES)

for i in range(0, N_FRAMES): 
    camera['position'] = jnp.dot(R, camera['position'])
    camera['angles'][0] += delta_degree
    ecran = engine.generation_image(camera, taille_ecran, resolution, liste_sphere, liste_lumiere, coefficients_lumiere, N_ECHANTILLON_STOCHASTIQUE)

    plt.figure()
    plt.imshow(ecran)
    plt.axis('off')
    plt.savefig(f"media/frames/frame_{i+1}.png", dpi=300)
    plt.close()

    print(f"Avancement : {i+1} / {N_FRAMES}", end="\r")

frames = jnp.stack([iio.imread(f"media/frames/frame_{i+1}.png") for i in range(0, N_FRAMES)], axis=0)
iio.mimwrite(f'media/video/animation{FORMAT}', frames, fps=FPS)

print("\nTravail terminé")