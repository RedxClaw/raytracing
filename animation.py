import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.animation as animation

import matplotlib.pyplot as plt
from module import setup, engine

fig = plt.figure()

camera, resolution, taille_ecran = setup.get_settings("480p")
liste_sphere, liste_lumiere = setup.get_objects()

def update(i):
    camera=([12, 0, 7],4, [180, -30*i] )
    ecran = engine.generation_image(camera, taille_ecran, resolution, liste_sphere, liste_lumiere)
    
    return ecran 

ani = animation.FuncAnimation(fig , update, frames=100, interval=1, blit=True)
plt.show()