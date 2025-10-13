import jax
from jax import jit, vmap, lax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from functools import partial


def pixels_to_global (pos_pixels, A, B, C, largeur_pixels, hauteur_pixels):

    t1= (1/largeur_pixels)*pos_pixels[0]
    t2= (1/hauteur_pixels)*pos_pixels[1]
    
    pos_global= np.zeros(3)

    pos_global[0]= A[0] * (1-t2-t1) + t1*B[0] + t2*C[0]
    pos_global[1] =  A[1] * (1-t2-t1) + t1*B[1] + t2*C[1] 
    pos_global[2] = A[2] * (1-t2-t1) + t1*B[2] + t2*C[2] 

    return pos_global 


def calcul_delta(pos_pt_focal, v, d, r):
    
    b= (2*np.dot(v, pos_pt_focal)-2*np.dot(d, v))
    a= np.dot(v, v)
    c= (np.dot(pos_pt_focal - d, pos_pt_focal - d)-r**2)

    scalaire= (b*b)-(4*a*c)
    return scalaire


def solution_intersection(pos_pt_focal, v, d, r, pos_pixels, A, B, C, largeur_pixels, hauteur_pixels):
    solution= []

    b=2*np.dot(v, pos_pt_focal - d)
    a=np.dot(v, v)
    delta = calcul_delta(pos_pt_focal, v, d, r)

    if delta==0: 
        #if((-b/(2*a))>=0):
            solution.append(-b/(2*a))
       # else:
            #solution=None

    elif delta > 0: 
        sol1= (-b+np.sqrt(delta))/(2*a)
        sol2= (-b-np.sqrt(delta))/(2*a)

        #if(sol1>=0):
           # solution.append(sol1)
           # print("test")
       # elif(sol2>=0):
           # solution.append(sol2)
        #else:
           # solution=None 
    elif(delta<0): 
        solution= None
    return solution


def couleur_pt(pos_pt_focal, liste_spheres, pos_pixels, A, B, C, largeur_pixels, hauteur_pixels):
    v=pixels_to_global(pos_pixels, A, B, C, largeur_pixels, hauteur_pixels)

    solutions_spheres=[]

    for sphere in liste_spheres: 
        #print(solution_intersection(pos_pt_focal,v, sphere[0], sphere[1], pos_pixels, A, B, C, largeur_pixels, hauteur_pixels))
        if (solution_intersection(pos_pt_focal,v, sphere[0], sphere[1], pos_pixels, A, B, C, largeur_pixels, hauteur_pixels)!=None):
            solutions_spheres.append(solution_intersection(pos_pt_focal,v, sphere[0], sphere[1], pos_pixels, A, B, C, largeur_pixels, hauteur_pixels))
        else:
            return 0
        
    minimum= solutions_spheres.index(min(solutions_spheres))
    couleur= liste_spheres[minimum][2]
    print(couleur)
    return couleur


#caractéristiques écran:         
largeur_pixel= 256
hauteur_pixel= 144

#caractéristiques caméra
x_cam = 8
y_cam = 0
z_cam = 0
angle_hor_cam = 0
angle_ver_cam = 0  
distance_focale= 8   
pos_pt_focal= np.array([2,0,0])
pos_cam= (x_cam, y_cam, z_cam, angle_hor_cam, angle_ver_cam)    

#Quatre coins écran
A= (1, -8 , 4.5)
B=(1, 8 , 4.5)
C=(1, -8, -4.5)
D=(1, 8, -4.5)


sphere= (np.array([0,0,0]), 1, 1) #centre, rayon, couleur
sphere2=(np.array([0.2,0,0]), 1.5, 3)

liste_spheres=[sphere, sphere2]

couleur_ecran= np.empty((largeur_pixel, hauteur_pixel))

for i in range (0, largeur_pixel):
    for j in range(0, hauteur_pixel):
        couleur_ecran[i][j]= couleur_pt(pos_pt_focal,liste_spheres, [i,j], A, B, C, largeur_pixel, hauteur_pixel)


plt.figure()
plt.imshow(couleur_ecran, cmap='inferno')
plt.show()