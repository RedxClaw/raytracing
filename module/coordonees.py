import numpy as np
from math import cos, sin, radians, acos, asin, degrees

def angle_to_cartesian(theta, phi):
    theta = radians(theta)
    phi = radians(phi)

    c_phi = cos(phi)
    return np.array([c_phi*cos(theta), c_phi*sin(theta), sin(phi)])

def cartesian_to_angle(p, sphere_c):
    p = p - sphere_c

    x = p[0]
    y = p[1]
    z = p[2]

    theta_s = acos(z/np.linalg.norm(p))
    phi_s = np.sign(y)*acos(x/np.linalg.norm(np.array([x, y])))

    return (degrees(theta_s), degrees(phi_s))


def get_corners(camera, taille_ecran):
    largeur = taille_ecran[0]
    hauteur = taille_ecran[1]

    vec_1 = angle_to_cartesian(camera[2][0], camera[2][1])
    vec_2 = np.array([-vec_1[1], vec_1[0], 0])
    vec_3 = np.cross(vec_1, vec_2)

    vec_1 = vec_1/np.linalg.norm(vec_1)
    vec_2 = vec_2/np.linalg.norm(vec_2)
    vec_3 = vec_3/np.linalg.norm(vec_3)

    ul = camera[0] + vec_1*camera[1] - largeur*vec_2/2 + hauteur*vec_3/2
    ur = camera[0] + vec_1*camera[1] + largeur*vec_2/2 + hauteur*vec_3/2
    dl = camera[0] + vec_1*camera[1] - largeur*vec_2/2 - hauteur*vec_3/2

    return (ul, ur, dl)

def local_to_global(i, j, corners, resolution):
    t1 = i/resolution[0]
    t2 = j/resolution[1]

    x = (1 - t1 - t2) * corners[0] + t1*corners[2] + t2*corners[1]

    return x


def taille_pixel(resolution): 
    hauteur_pix = 9/resolution[0]
    largeur_pix = 16/resolution[1]

    return hauteur_pix, largeur_pix
