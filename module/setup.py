import json
import jax.numpy as jnp
from jax import lax

from module.texture import *

Textures = {
    "unicolore": unicolore,
    "damier": damier,
    "spirale": spirale
}

Couleurs = {
    "noir": [0.1, 0.1, 0.1],
    "rouge": [1, 0, 0],
    "orange": [1, 0.5, 0],
    "jaune": [1, 1, 0],
    "vert": [0, 1, 0],
    "bleu": [0, 0, 1],
    "magenta": [1, 0, 1],
    "cyan": [0, 1, 1],
    "rose": [1, 0, 0.5],
    "lime": [0.5, 1, 0],
    "turquoise": [0, 1, 0.5],
    "mauve": [0.5, 0, 1],
    "blanc": [1, 1, 1]
}

liste_textures = list(Textures.keys())
liste_ft_textures = list(Textures.values())

liste_couleurs = list(Couleurs.keys())
liste_resolution = ["144p", "360p","480p", "720p", "1080"]

"""
    get_settings(resolution)

Extrait, depusi un fichier JSON, l'ensemble des données générales nécessaires au bon fonctionnement du code

# INPUT :
* `hauteur_resolution`      Nombre positif représentant la hauteur de l'écran qui sera simulé

# OUTPUT :
* `camera`          Dictionnaire contenant :
    * `'position'`          Le vecteur position du point focal
    * `'distance_focale'`   La distance entre le point focal et l'écran simulé
    * `'angles'`            Paire d'angles permettant d'obtenir l'orientation de la caméra
* `resolution`      Paire d'entiers positifs donnant la résolution de l'écran simulé
* `taille_ecran`    Paire de réels qui donne la taille de l'écran simulé
"""
def get_settings(fichier_parametres, hauteur_resolution):
    with open(fichier_parametres, 'r') as file:
        data_settings = json.load(file)
        
    cam_pos = jnp.array(data_settings["camera"]["position"])
    cam_focal = data_settings["camera"]["distance_focale"]
    cam_angles = [data_settings["camera"]["angles"][0], data_settings["camera"]["angles"][1]]

    camera = {
        "position": cam_pos,
        "distance_focale": cam_focal,
        "angles": cam_angles
    }
    
    taille_ecran = {'largeur': data_settings["ecran"]["largeur"],
                    'hauteur': data_settings["ecran"]["hauteur"]}

    if type(hauteur_resolution) == int and hauteur_resolution > 0:
        resolution = (round(hauteur_resolution), round(taille_ecran['largeur']*hauteur_resolution/taille_ecran['hauteur']))
    else:
        print("""Erreur : La résolution choisie n'est pas valide. Veuillez choisir un entier positif qui correspondra au nombre de pixels de la hauteur de l'écran. \n""")
        exit(1)
    
    coefficients_lumiere = list(data_settings['diffusion_lumiere'].values())

    return camera, resolution, taille_ecran, coefficients_lumiere

"""Extrait depuis un fichier JSON l'ensemble des données des objets de la scène"""
def get_objects(fichier_objets):
    with open(fichier_objets, 'r') as file:
        data = json.load(file)

    Dictionnaire_Sphere = {
        "position": [],
        "rayon": [],
        "motif": [],
        "couleurs": [],
        "metallicite": []
    }

    Dictionnaire_Lumiere = {
        "position": [],
        "intensite": [],
        "couleur": []
    }

    for categorie in data:
        liste_objet = data[categorie]
        if categorie == "sphere":
            for sphere in liste_objet:
                Dictionnaire_Sphere["position"].append(jnp.array(sphere["position"]))

                Dictionnaire_Sphere["rayon"].append(sphere["rayon"])

                motif = sphere["texture"]["motif"]
                if not (motif in liste_textures):
                    print("Erreur : Le motif choisi n'est pas valide. Veuillez choisir parmi les motifs suivants :", liste_textures)
                    exit(1)
                
                Dictionnaire_Sphere["motif"].append(liste_textures.index(motif))

                Dictionnaire_Sphere["metallicite"].append(max(sphere["metallicite"], 0))

                couleurs_sphere = []
                for couleur in sphere["texture"]["couleurs"]:
                    if not couleur in liste_couleurs:
                        print("Erreur : La couleur -", couleur, "- choisie n'est pas valide. Veuillez choisir parmi les couleurs suivantes :", liste_couleurs)
                        exit(1)
                    couleurs_sphere.append(jnp.array(Couleurs[couleur]))
                Dictionnaire_Sphere["couleurs"].append(couleurs_sphere)

        elif categorie == "lumiere":
            for lumiere in liste_objet:
                Dictionnaire_Lumiere["position"].append(jnp.array(lumiere["position"]))
                Dictionnaire_Lumiere["intensite"].append(lumiere["intensite"])
                if type(lumiere["couleur"]) == str:
                    Dictionnaire_Lumiere["couleur"].append(jnp.array(Couleurs[lumiere["couleur"]]))
                else:
                    Dictionnaire_Lumiere["couleur"].append(jnp.array(lumiere["couleur"]))

    Dictionnaire_Sphere["position"]     = jnp.array(Dictionnaire_Sphere["position"])
    Dictionnaire_Sphere["rayon"]        = jnp.array(Dictionnaire_Sphere["rayon"])
    Dictionnaire_Sphere["motif"]        = jnp.array(Dictionnaire_Sphere["motif"])
    Dictionnaire_Sphere["couleurs"]     = jnp.array(Dictionnaire_Sphere["couleurs"])
    Dictionnaire_Sphere["metallicite"]  = jnp.array(Dictionnaire_Sphere["metallicite"])

    Dictionnaire_Lumiere["position"]    = jnp.array(Dictionnaire_Lumiere["position"])
    Dictionnaire_Lumiere["intensite"]   = jnp.array(Dictionnaire_Lumiere["intensite"])
    Dictionnaire_Lumiere["couleur"]     = jnp.array(Dictionnaire_Lumiere["couleur"])

    return Dictionnaire_Sphere, Dictionnaire_Lumiere

"""
    setup_ecran(resolution)

Crée un tableau contenant dans chaque case les indices de la case dans le tableau (au sens matriciel)

# INPUT :
* `resolution`  La résolution de l'écran simulé

# OUTPUT :
* `tableau`     Une matrice contenant dans chaque case les indices correspondants
"""
def setup_ecran(resolution):
    x = jnp.linspace(start=0, stop=resolution[0], num=resolution[0]+1)
    y = jnp.linspace(start=0, stop=resolution[1], num=resolution[1]+1)
    X, Y = jnp.meshgrid(x, y, indexing="xy")
    tableau = jnp.array([X, Y])
    return tableau           