import json
import numpy as np
from math import fmod

import jax
from jax import jit, vmap, lax
import jax.numpy as jnp
from functools import partia

from module import coordonees

def min_positif(a, b):
    if a > b and b > 0:
        return b
    return a



def intersection_sphere(x, v, liste_sphere, ignore_iteration):
    iteration = len(liste_sphere)
    alpha = 65535
    for i in range(iteration):
        if (i == ignore_iteration):
            continue
        sphere = liste_sphere[i]
        d = sphere[0] #Position de la sphere
        r = sphere[1] #Rayon

        a = np.dot(v, v)
        b = 2*np.dot(v, x-d)
        c = np.dot(x - d, x - d) - r*r
        
        delta = b*b - 4*a*c

        if delta == 0:
            if alpha != min_positif(alpha, -b/(2*a)):
                iteration = i
            alpha = min_positif(alpha, -b/(2*a))
        elif delta > 0:
            s1 = (-b - np.sqrt(delta))/(2*a)
            s2 = (-b + np.sqrt(delta))/(2*a)
            if alpha != min_positif(min_positif(alpha, s1), min_positif(alpha, s2)):
                iteration = i
            alpha = min_positif(min_positif(alpha, s1), min_positif(alpha, s2))
        
    if alpha == 65535:
        alpha = 0 
        iteration = len(liste_sphere)
    return alpha, iteration
            
def vecteurs_lumiere(pos_camera, p, pos_lumiere, pos_sphere):

    n0 = p - pos_sphere 
    vl0 = pos_lumiere - p 
    vc0 = pos_camera - p  
    vr0 = 2*np.dot(vl0, n0)*n0 - vl0
        
    if np.dot(n0, n0)==0 or np.dot(vl0, vl0)==0 or np.dot(vc0, vc0)==0 or np.dot(vr0, vr0)==0:
        return 0
    
    vl = vl0/np.linalg.norm(vl0)
    n = n0/np.linalg.norm(n0)
    vc = vc0/np.linalg.norm(vc0)
    vr = vr0/np.linalg.norm(vr0)

    return n, vl, vc, vr

def calcul_lumiere(camera, p, liste_lumiere, liste_sphere, iteration_sphere, alpha):
    sphere = liste_sphere[iteration_sphere]
    k_a = 0.4
    k_d = 0.5
    k_s = 0.6
    beta = sphere[3]

    intensite_total = np.zeros(3)
    for lumiere in liste_lumiere:
        intensite = 40*lumiere[1]/np.dot(p - lumiere[0], p - lumiere[0])
        n, vl, vc, vr = vecteurs_lumiere(camera[0], p, lumiere[0], sphere[0])

        alpha, iteration = intersection_sphere(p, lumiere[0] - p, liste_sphere, iteration_sphere)

        factor = k_a * intensite

        if not (0 < alpha < 1):
            factor += k_d * np.dot(vl, n) * intensite + k_s * (np.dot(vr, vc)**beta) * intensite

        intensite_total += factor*lumiere[2]
    
    for i in range(3):
        if intensite_total[i] > 1: intensite_total[i] = 1
    
    return intensite_total

def moyenne_lumiere(camera, liste_lumiere, liste_sphere, iteration_sphere, resolution, pos_pixel_x, pos_pixel_y, alpha, corners): 
    hauteur_pixel, largeur_pixel = coordonees.taille_pixel(resolution) 
    e_x= np.random.uniform (-largeur_pixel/2, largeur_pixel/2, 20)
    e_y = np.random.uniform (-hauteur_pixel/2, hauteur_pixel/2, 20) 
    lum=0

    for i in range(0,20):
        e = coordonees.local_to_global(pos_pixel_x + e_x[i] , pos_pixel_y + e_y[i], corners, resolution)
        v= e-camera[0]

        lum=lum+calcul_lumiere(camera, e + alpha*v, liste_lumiere, liste_sphere, iteration_sphere, alpha) 

    lum=lum/20
    return lum
