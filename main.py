import matplotlib.pyplot as plt

from module import setup, engine

camera, resolution, taille_ecran = setup.get_settings("360p")
liste_sphere, liste_lumiere = setup.get_objects()

ecran = engine.generation_image(camera, taille_ecran, resolution, liste_sphere, liste_lumiere)

plt.figure()
plt.imshow(ecran[:, :, 2:5])
plt.axis('off')
plt.show()