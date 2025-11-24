from tkinter import *

def init_fenetre():
    root = Tk()
    root.title("Configurateur de la scène")
    root.geometry("500x320")
    photo = PhotoImage(file="assets\\icons\\main.png")
    root.wm_iconphoto(False, photo)

    objets = Frame(root, relief=RAISED, height=max).pack(side=LEFT)
    sphere = Frame(objets, relief=RAISED).pack(side=TOP)

    label_sphere_txt = StringVar()
    label_sphere_txt.set("Sphères")
    label_sphere = Label(sphere, textvariable=label_sphere_txt, font=("Arial", 16, "bold")).pack()
    
    lumiere = Frame(objets, relief=RAISED).pack(side=BOTTOM)


    apercu = Frame(root, relief=RAISED, height=max).pack(side=RIGHT)
    upper_apercu = Frame(apercu, relief=FLAT).pack(side=TOP)
    lower_apercu = Frame(apercu, relief=FLAT).pack(side=BOTTOM)

    Button(lower_apercu, text="Annuler", command=exit).pack(side=RIGHT)
    Button(lower_apercu, text="Générer", command=root.destroy).pack(side=RIGHT)

    return root