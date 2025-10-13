import jax
from jax import jit, vmap, lax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

#IDEE calcul distance difference de vecteur 
def distance(v1, v2):
    v = v1 - v2
    d = np.dot(v)
    return d

#IDEE
def max_distance(liste_spheres, pos_pt_focal):
    max=0
    for i in range(0, np.size(liste_spheres, 2)):
        if distance(pos_pt_focal, liste_spheres[max]["position"]) >= distance(pos_pt_focal, liste_spheres[i+1]["position"]):
            max=max
        else: 
            max=i
    
    return max 


def pixels_to_global (pos_pixels, A, B, C, largeur_pixels, hauteur_pixels):
    d_t1= 1/largeur_pixels
    d_t2= 1/hauteur_pixel

    t1= d_t1*pos_pixels[0]
    t2= d_t2*pos_pixels[1]
    
    pos_global= np.zeros(3)

    pos_global[0]= A[0] * (1-t2-t1) + t1*B[0] + t2*C[0]
    pos_global[1] =  A[1] * (1-t2-t1) + t1*B[1] + t2*C[1] 
    pos_global[2] = A[2] * (1-t2-t1) + t1*B[2] + t2*C[2] 

    return pos_global 

def solution_sphere(pos_pt_focal, v, d, r):
    scalaire=( (np.dot(pos_pt_focal, v))**2 - 2*(np.dot(pos_pt_focal,d))*(np.dot(v, d)) + (np.dot(v, d))**2 -(np.dot(v, v))* ((np.dot(pos_pt_focal, pos_pt_focal))+ 2* (np.dot(pos_pt_focal, d))+np.dot(d, d)-r**2))
    return scalaire


#def verif_intersection2 (pos_pixels, pos_pt_focal, A, B, C, largeur_pixels, hauteur_pixels, liste_spheres):
 #   i=0
 #   v=pixels_to_global(pos_pixels, A, B , C, largeur_pixels, hauteur_pixels)
 #   while i<=np.size(liste_spheres): ou tant que ensemble non vide 
 #       max=max_distance(liste_spheres, pos_pt_focal)
 #       if (solution_sphere(pos_pt_focal, v,liste_spheres[max]["position"], liste_spheres[max]["rayon"])>=0):
 #           return liste_spheres[max]["couleur"]
 #      
    #travailler avec ensemble et non liste de sphère ? 



def verif_intersection(pos_pixels, pos_pt_focal, A, B, C, largeur_pixels, hauteur_pixels, sphere, sphere2):
    v=pixels_to_global(pos_pixels, A, B , C, largeur_pixels, hauteur_pixels)

    d=sphere[0]
    d2=sphere2[0]

    r=sphere[1]
    r2=sphere2[1]

    c=sphere[2]
    c2=sphere2[2]

    if ((np.dot(pos_pt_focal, d))>= np.dot(pos_pt_focal,d2)):
        #if ( (np.dot(pos_pt_focal, v))**2 - 2*(np.dot(pos_pt_focal,d))*(np.dot(v, d)) + (np.dot(v, d))**2 -(np.dot(v, v))* ((np.dot(pos_pt_focal, pos_pt_focal))+ 2* (np.dot(pos_pt_focal, d))+np.dot(d, d)-r**2)) >=0:
        if (solution_sphere(pos_pt_focal, v, d, r) >=0):    
            return c
        elif(solution_sphere(pos_pt_focal, v, d2, r2) >=0):
            return c2
        else: 
            return 0
        
    else: 
        if(solution_sphere(pos_pt_focal, v, d2, r2) >=0):
            return c2
        elif (solution_sphere(pos_pt_focal, v, d, r) >=0):    
            return c
        else:
            return 0

    

#caractéristiques écran:         
largeur_pixel = 640
hauteur_pixel = 360

#caractéristiques caméra
x_cam = 8
y_cam = 0
z_cam = 0
angle_hor_cam = 0
angle_ver_cam = 0  
distance_focale = 8   
pos_pt_focal = np.array([2,0,0])
pos_cam = (x_cam, y_cam, z_cam, angle_hor_cam, angle_ver_cam)    

#Quatre coins écran
A = (1, -8 , 4.5)
B = (1, 8 , 4.5)
C = (1, -8, -4.5)
D = (1, 8, -4.5)


sphere = (np.array([0, 0, 0]), 1, 1) #centre, rayon, couleur
sphere2 = (np.array([0.5, 0, 0.5]), 1.8, 3)

#sphere1={"position": [0,0,0], "rayon":1, "couleur":1}
#sphere2= {"position": [0.5,0,0.5], "rayon":1.8, "couleur":3}
#liste_spheres=(sphere1, sphere2)


couleur_ecran = np.empty((largeur_pixel, hauteur_pixel))

for i in range (0, largeur_pixel):
    for j in range(0, hauteur_pixel):
        couleur_ecran[i][j]= verif_intersection((i,j), pos_pt_focal, A, B, C, largeur_pixel, hauteur_pixel, sphere, sphere2)

plt.figure()
plt.imshow(couleur_ecran, cmap='inferno')
plt.show()


#Pour lundi: couleurs + réfléchir au cas à deux sphères 
#tuples pour les sphères