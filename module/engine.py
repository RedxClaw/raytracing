import numpy as np
from module import coordonees, raytracing, setup, spirale


def generation_image(camera, taille_ecran, resolution, liste_sphere, liste_lumiere):
    ecran = setup.setup_ecran(resolution)
    corners = coordonees.get_corners(camera, taille_ecran)
    for i in range(resolution[0]):
        for j in range(resolution[1]):
            x = coordonees.local_to_global(i, j, corners, resolution)
            v = x - camera[0]

            alpha, iteration = raytracing.intersection_sphere(x, v, liste_sphere, len(liste_sphere))

            pos_S= x+alpha*v 

            if iteration != len(liste_sphere):
                theta_s, phi_s = coordonees.cartesian_to_angle(pos_S - liste_sphere[iteration][0], liste_sphere[iteration][1])
                couleur = spirale.spirale(theta_s, phi_s, liste_sphere[iteration][2]) #condition ou fonction
                intensite = raytracing.moyenne_lumiere(camera, liste_lumiere, liste_sphere, iteration, resolution, i, j, alpha, corners)
                #intensite = raytracing.calcul_lumiere(camera, x + alpha*v, liste_lumiere, liste_sphere, iteration, alpha)
            else:
                couleur = np.array([0, 0, 0])
                intensite = np.zeros(3)

            ecran[i, j, 2:5] = intensite*couleur
    
    return ecran
