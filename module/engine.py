import numpy as np
from module import coordonees, raytracing, setup

def generation_image(camera, taille_ecran, resolution, liste_sphere, liste_lumiere):
    ecran = setup.setup_ecran(resolution)
    corners = coordonees.get_corners(camera, taille_ecran)
    for i in range(resolution[0]):
        for j in range(resolution[1]):
            x = coordonees.local_to_global(i, j, corners, resolution)
            v = x - camera[0]

            alpha, iteration = raytracing.intersection_sphere(x, v, liste_sphere, len(liste_sphere))

            if iteration != len(liste_sphere):
                sphere = liste_sphere[iteration]
                theta_s, phi_s = coordonees.cartesian_to_angle(x + alpha*v, sphere[0])
                couleur = sphere[2]["motif"](theta_s, phi_s, sphere[2]["couleurs"])
                intensite = raytracing.calcul_lumiere(camera, x + alpha*v, liste_lumiere, liste_sphere, iteration)
            else:
                couleur = np.array([0, 0, 0])
                intensite = np.zeros(3)

            ecran[i, j, 2:5] = intensite*couleur
    
    return ecran