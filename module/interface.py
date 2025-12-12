from tkinter import *
from tkinter import ttk

def print_input():
    sphere= nbre_sphere_entry.get()
    print(sphere)
    return sphere

root=Tk()
root.title("Interface objets")

root.geometry('900x600+50+50')
root.resizable(0,0)

root.attributes('-topmost', 1)

Label(root, text = 'Choix des caractéristiques').pack()

titre_nbre_sphere_entry = Label(root, text='Nombre de sphères:')
titre_nbre_sphere_entry.place(x=400, y=200)

nbre_sphere = StringVar()
nbre_sphere_entry = Entry(root, textvariable=nbre_sphere)
nbre_sphere_entry.place(x=395, y=230)
nbre_sphere_entry.focus()

confirmation_sphere=Button(root, text="Confirmer", command=print_input)
confirmation_sphere.place(x=423, y=260)



exit_button = ttk.Button( root,text='Exit',command=lambda: root.quit())
exit_button.pack(ipadx=100,ipady=5,)
exit_button.place(x=800, y= 550)


root.mainloop()
