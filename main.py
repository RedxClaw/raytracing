import matplotlib.pyplot as plt

from module import setup, engine
from os import path, mkdir
from shutil import rmtree

FICHIER_PARAMETRES          = "./module/json/settings.json"
FICHIER_OBJETS              = "./module/json/objets_1.json"

RESOLUTION_HAUTEUR          = 720
N_ECHANTILLON_STOCHASTIQUE  = 20

if not path.isdir('media'):
    mkdir('media')
if path.isdir('media/images'):
    rmtree('media/images')
mkdir('media/images')

camera, resolution, taille_ecran, coefficients_lumiere = setup.get_settings(FICHIER_PARAMETRES, RESOLUTION_HAUTEUR)
liste_sphere, liste_lumiere = setup.get_objects(FICHIER_OBJETS)

ecran = engine.generation_image(camera, taille_ecran, resolution, liste_sphere, liste_lumiere, coefficients_lumiere, N_ECHANTILLON_STOCHASTIQUE)

plt.figure()
plt.imshow(ecran)
plt.axis('off')
plt.savefig("media/images/raytracing.png", dpi=300)
plt.show()