import pandas as pd

Datos = pd.read_csv('carros.csv')
Datos.head(22)

features_train = Datos.iloc[0:22, 0:22]
target_train = Datos.iloc[0:22, 22]

#Import LabelEncoder
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
f0 = le.fit_transform(features_train.iloc[0:22, 0])
f1 = le.fit_transform(features_train.iloc[0:22, 1])
f2 = le.fit_transform(features_train.iloc[0:22, 2])
f3 = le.fit_transform(features_train.iloc[0:22, 21])
label = le.fit_transform(target_train)
features = list(zip(f0,f1,f2,f3))
print(features)
print(label)


from sklearn.naive_bayes import GaussianNB
model1 = GaussianNB()
model1.fit(features, label)
#Predicted = model1.predict([[1,1,0,2]])
#Predicted = model1.predict([[0,1,2,1]])
#Predicted = model1.predict([[0,0,2,1]])
Predicted = model1.predict([[0,0,0,2]])
print(Predicted)


le.inverse_transform([2])


from sklearn.naive_bayes import GaussianNB
def bayesiano(num1, num2, num3, num4):
    #model1 = GaussianNB()
    #model1.fit(features, label)
    return model1.predict([[num1,num2,num3,num4]])


import tkinter as tk
from tkinter import ttk

raiz = tk.Tk()

raiz.title("Autos Robados")
#raiz.resizable(False, False)
raiz.geometry("650x600")

miFrame = tk.Frame(raiz, width="500", height="500")
miFrame.pack(anchor="center", expand="True")
miFrame.config(bg="red")
labelTop = tk.Label(miFrame, text = "Seleccione Modelo Auto")
labelTop.grid(column=0, row=0)

comboMarca = ttk.Combobox(miFrame, 
                            values=[
                                    "A2", 
                                    "Metro",
                                    "Mini",
                                    "Kalos",
                                    "Matiz",
                                    "2 CV",
                                    "AX",
                                    ])
                                    
comboMarca.grid(column=0, row=1)
comboMarca.current(1)

comboModelo = ttk.Combobox(miFrame, 
                            values=[
                                    "Min_Smin_Audi", 
                                    "Min_Smin_Austin/Morris",
                                    "Min_Smin_Chevrolet",
                                    "Sm_Sal_Volkswagen",
                                    "Med_Sal_Audi,",
                                    "2 CV",
                                    "AX",
                                    ])
                                    
comboModelo.grid(column=0, row=1)
comboModelo.current(1)

boton = tk.Button(raiz, text="Evaluar Carro proveble a Robar?", command=bayesiano)
boton.pack()


def bayesiano(num1, num2, num3, num4):
    #model1 = GaussianNB()
    #model1.fit(features, label)
    return model1.predict([[num1,num2,num3,num4]])

raiz.mainloop()