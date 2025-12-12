from math import fmod

def unicolore(theta_s, phi_s, couleurs):
    return couleurs[0]

def damier (theta_s, phi_s, couleurs):
    if 0<=fmod(abs(theta_s), 10) <=4 :
        if 4<=fmod(abs(phi_s),10) <=9 :
            return couleurs[1]
    else :
        if 0<=fmod(abs(phi_s), 10)<= 4 :
            return couleurs[1]
        
    return couleurs[0]

def spirale(theta_s, phi_s, couleurs):
    if 0 < fmod(abs(theta_s+phi_s), 20)<=10:
        return[couleurs[1]]

    return couleurs[0]