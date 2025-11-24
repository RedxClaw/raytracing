import json
import numpy as np
from module import textures

Couleurs = {
    "noir": [0, 0, 0],
    "rouge": [1, 0, 0],
    "orange": [1, 0.5, 0],
    "jaune": [1, 1, 0],
    "vert": [0, 1, 0],
    "bleu": [0, 0, 1],
    "magenta": [1, 0, 1],
    "rose": [1, 0, 0.5],
    "lime": [0.5, 1, 0],
    "turquoise": [0, 1, 0.5],
    "mauve": [0.5, 0, 1],
    "blanc": [1, 1, 1]
}

Textures = {
    "damier" : textures.damier,
    "spirale": textures.spirale,
    "unicolore": textures.unicolore
}

def get_settings(resolution):
    with open('module\\json\\settings.json', 'r') as file:
        data_settings = json.load(file)
        
    cam_pos = np.array(data_settings["camera"]["position"])
    cam_focal = data_settings["camera"]["distance_focale"]
    cam_angles = (data_settings["camera"]["angles"][0], data_settings["camera"]["angles"][1])

    camera = (cam_pos, cam_focal, cam_angles)

    resolution = data_settings["resolution"][resolution]

    taille_ecran = (data_settings["ecran"]["largeur"], data_settings["ecran"]["hauteur"])

    return camera, resolution, taille_ecran

def setup_ecran(resolution):
    ecran = np.zeros([resolution[0], resolution[1], 5])

    for i in range(resolution[0]):
        for j in range(resolution[1]):
            ecran[i, j, 0] = i
            ecran[i, j, 1] = j
    
    return ecran

def get_color(couleur):
    if type(couleur) == str:
        return np.array([Couleurs[couleur]])
    return couleur

def get_objects():
    with open('module\\json\\objects.json', 'r') as file:
        data = json.load(file)

    liste_sphere = []
    liste_lumiere = []

    for categorie in data:
        liste_objet = data[categorie]
        if categorie == "sphere":
            for dico in liste_objet:
                sphere_pos = np.array(dico["position"])
                sphere_rayon = dico["rayon"]

                sphere_metalicite = dico["metalicite"]
                sphere_texture = {"motif": Textures[dico["texture"]["motif"]], "couleurs": []}
                for couleur in dico["texture"]["couleurs"]:
                    sphere_texture["couleurs"].append(get_color(couleur))

                liste_sphere.append((sphere_pos, sphere_rayon, sphere_texture, sphere_metalicite))

        elif categorie == "lumiere":
            for dico in liste_objet:
                lumiere_pos = np.array(dico["position"])
                lumiere_intensite = dico["intensite"]
                if type(dico["couleur"]) == str:
                    lumiere_couleur = np.array(Couleurs[dico["couleur"]])
                else:
                    lumiere_couleur = np.array(dico["couleur"])

                liste_lumiere.append((lumiere_pos, lumiere_intensite, lumiere_couleur))
                
    return liste_sphere, liste_lumiere     


def taille_pixel(resolution):
    hauteur_pix = 9/resolution[0]
    largeur_pix = 16/resolution[1]

    return hauteur_pix, largeur_pix