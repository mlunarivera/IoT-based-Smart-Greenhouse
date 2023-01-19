# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 20:40:43 2020
Código para predicción de temperatura utilizando ANN y RNN

@author: Carlos A. Hernández
"""
#Librerias para manejo de datos
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
#Librerias para redes neuronales
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten, LSTM, SimpleRNN
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler

"""Entrega una lista de datos normalizados en un rango definido
   recibe como datos de entrada: Una lista de datos 't', """
def Normalizar(t, scaler):
    #La función de escalamiento no acepta arreglos 1D
    scaled = scaler.fit_transform(t.reshape(-1,1))#Normalización
    return scaled.reshape(-1)#Regresa el vector normalizado en 1D

"""Crear modelos ANN"""
def ANN(pasos, neuronas):
    model = Sequential()
    model.add(Dense(neuronas, init='normal', input_shape=(1,pasos), activation = 'tanh'))#Primer capa oculta
    model.add(Flatten())
    model.add(Dense(1, init='normal', activation='linear'))# Capa de salida
    opt = optimizers.SGD(lr=0.012, momentum=0.7, nesterov=True) #Parametros del SGD
    model.compile(loss = 'mse', optimizer='sgd', metrics=['accuracy'])
    model.summary()
    return model

def RNN(pasos, neuronas):
    model = Sequential()
    #model.add(TimeDistributed(Flatten()))
    model.add(LSTM(neuronas, init='normal', input_shape=(1,pasos) ))#Primer capa oculta
    #model.add(Flatten())
    model.add(Dense(1, init='normal', activation='linear'))#Capa de saliada
    optr = optimizers.SGD(lr=0.015, momentum=0.7, nesterov=True)
    model.compile(loss='mse', optimizer=optr, metrics=['accuracy'])
    model.summary()
    return model

"""Entrega una lista de diccionarios con los datos separados por fechas"""
def CargarDatos(archivo, scaler):
    fechas = []
    datos = pd.read_csv(archivo, delimiter="\s+", parse_dates=[], header=None, squeeze=True, names=['fecha','hora','temperatura', 'humedad', 'co2'])
    datos.head()
    #obteniendo fechas 
    for i in datos['fecha']:
        if i not in fechas:
            fechas.append(i)
    fechas.reverse()
    #Normalización de datos
    auxT = Normalizar(datos['temperatura'][::-1].values.astype(float), scaler)
    datos['temperatura'] = auxT#[::-1]#Normalizar(datos['temperatura'].values.astype(float),scaler)#Normalizar(datos['temperatura'].values.astype(float),scaler)
    #Construllendo los diccionarios de la lista a enviar
    sdata = [] #lista de diccionarios
    for i in fechas:
        d = datos[datos.fecha == i] #datos por fecha
        dct = dict() #diccionario temporal 
        dct['Fecha'] = i
        dct['temperatura'] = d['temperatura']#Normalizar(d['temperatura'][::-1].values.astype(float),scaler) #Datos normalizados
        dct['Cantidad'] = len(d)
        sdata.append(dct) #Agregar diccionario a la lista
    return sdata

def PrediccionANN(data, model, scaler):
    result = model.predict(data)# Resultado de la predicción
    temp = [x for x in result]# Cambiar formato?
    temp=result.reshape(-1, 1)#Se van a denormalizar por lo que se debe cambiar de dimención
    inverted = scaler.inverse_transform(temp)
    return inverted 

"""Entrega estadisticos de MAE y RMSE"""
def Estadisticos(salida, real):
    #MAE
    mape = 0
    m2=np.zeros(len(real));
    e = abs(salida - real)
    mae = sum(e)/len(e)
    #RMSE
    temp = salida - real
    rmse = np.sqrt((sum(temp*temp))/len(temp))
    #MAPE
    m = abs(real-salida);
    for i in range(len(m-1)):
        m2[i] = m[i]/real[i]
    mape = (sum(m2)/len(m2))*100;
    return mae, rmse, mape

"""Código principal"""

#Variables para construir la red
inputs = 8
neuronas = 8
scaler = MinMaxScaler(feature_range=(-1,1)) #Se define los limites de la normalización
v = CargarDatos("Datos_ieee/Datos_03_M.txt", scaler)#Lista de diccionarios de los datos

#ventana para las graficas
plt.rcParams["figure.figsize"] = [20, 10]
plt.style.use('fast')

#Preparando los datos (Convertir la serie temporal a un dataset) para un problema supervisado
aux = v[0]['temperatura']
for i in range(1,(inputs+1)):
    aux = np.vstack((aux, v[i]['temperatura']))    
sup = aux.transpose()
#Construir matriz entera
for i in range(1, len(v)-inputs):
    aux = v[i]['temperatura']
    for j in range(1,(inputs+1)):
        aux = np.vstack((aux, v[i+j]['temperatura']))
    sup = np.vstack((sup, aux.transpose()))

#Definiendo datos de entrenamiento y validación
train_days = int(len(sup)*0.80)#numeo de conjuntos de datos de prueba (en total son 21024, se toma el 75% de datos para prueba)
train = sup[0:train_days] 
test = sup[train_days:]

#dividir los datos como entradas y salidas
x_train, y_train = train[:, :-1], train[:, -1]
x_val, y_val = test[:, :-1], test[:, -1]
#reshape
x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
x_val = x_val.reshape((x_val.shape[0], 1, x_val.shape[1]))

#Crear, entrenar y validar la ANN
mae2 = 0
rmse2= 0
mape2 = 0
iteraciones = 1
for f in range(iteraciones):
    model = ANN(inputs, neuronas)#crear modelo de la ANN
    #Entrenar y validar la ANN
    history = model.fit(x_train, y_train, epochs=40, validation_data=(x_val, y_val), batch_size=8, verbose = 1)#batch=25
    train_loss=history.history['loss']
    prediccion = PrediccionANN(x_val, model, scaler)#PrediccionANN(x_val, model, scaler)

    #Obtener estadisticos
    real = [x for x in y_val]
    real = y_val.reshape(-1, 1)
    real = scaler.inverse_transform(real)
    est = Estadisticos(prediccion, real)
    mae2 += est[0]
    rmse2 += est[1]
    mape2 += est[2]
    
mae2 = mae2/iteraciones 
rmse2 = rmse2/iteraciones
mape2 = mape2/iteraciones 

#Graficar el accuarecy
plt.figure()
plt.plot(history.history['accuracy'])
#Graficar el entrenamiento
plt.figure()
plt.plot(history.history['loss'],'r')
#plt.plot(history.history['val_loss'],'g')
#Gráfica de la predicción contra el real
plt.figure()
plt.plot(range(len(real[0:1050])),real[0:1050],c='g')#En color verde los valores reales (1300 - 2350 ok)
plt.plot(range(len(prediccion[0:1050])),prediccion[0:1050],c='r')#En color rojo los valores predichos

realmat = real[0:1050]
predicmat = prediccion[0:1050]

