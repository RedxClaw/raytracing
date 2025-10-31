import json
import numpy as np

Couleurs = {
    "noir": [0, 0, 0],
    "rouge": [1, 0, 0],
    "vert": [0, 1, 0],
    "bleu": [0, 0, 1],
    "blanc": [1, 1, 1]
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
                if type(dico["couleur"]) == str:
                    sphere_couleur = np.array(Couleurs[dico["couleur"]])
                else:
                    sphere_couleur = np.array(dico["couleur"])
                sphere_metalicite = dico["metalicite"]

                liste_sphere.append((sphere_pos, sphere_rayon, sphere_couleur, sphere_metalicite))

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

                    