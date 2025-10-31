import numpy as np

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
            
def vecteurs_lumiere(camera, p, lumiere, sphere):
    n = p - sphere[0]
    vl = lumiere[0] - p
    vc = camera[0] - p
    vr = 2*np.dot(vl, n)*n - vl

    n = n/np.linalg.norm(n)
    vl = vl/np.linalg.norm(vl)
    vc = vc/np.linalg.norm(vc)
    vr = vr/np.linalg.norm(vr)

    return n, vl, vc, vr

def calcul_lumiere(camera, p, liste_lumiere, liste_sphere, iteration_sphere):
    sphere = liste_sphere[iteration_sphere]
    k_a = 0.4
    k_d = 0.5
    k_s = 0.6
    beta = sphere[3]

    intensite_total = np.zeros(3)
    for lumiere in liste_lumiere:
        intensite = 40*lumiere[1]/np.dot(p - lumiere[0], p - lumiere[0])
        n, vl, vc, vr = vecteurs_lumiere(camera, p, lumiere, sphere)

        alpha, iteration = intersection_sphere(p, lumiere[0] - p, liste_sphere, iteration_sphere)

        factor = k_a * intensite

        if not (0 < alpha < 1):
            factor += k_d * np.dot(vl, n) * intensite + k_s * (np.dot(vr, vc)**beta) * intensite

        intensite_total += factor*lumiere[2]
    
    for i in range(3):
        if intensite_total[i] > 1: intensite_total[i] = 1
    
    return intensite_total
        