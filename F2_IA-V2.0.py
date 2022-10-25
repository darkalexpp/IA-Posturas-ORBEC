#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
import pandas as pd
import os
import numpy as np 
import sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical 
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# In[2]:


#Parametros configurables
porcentajeDataSetPruebas=0.2
semillaRandom=42    #42
columnaObjetivo='Postura_correcta'
pathDeTrabajo='./'


# In[3]:


#Leer datos
print('Importado datos del csv...')
datos = pd.read_csv(pathDeTrabajo+'Angulos.csv',sep=';', index_col=[0])
#Limpiar Datos
#datos = datos[datos.Postura_correcta != ' ']
#datos = datos[datos.A4 != '180']
#datos = datos[datos.A9 != '0']
#datos = datos[datos.A10 != '0']
datos = datos.round(3)
print('Tamanio del DataSet')
print(datos.shape)


# In[4]:


Col=['A1', 'A2', 'A3', 'A4', 'A5', 'A7', 'A8','A9','A10', 'A11']
# Extraemos el conjunto de clases.
classes = datos[columnaObjetivo].unique().tolist()
print(f'Clases: {classes}')

v=np.zeros(shape=[datos[columnaObjetivo].shape[0],len(classes)])
for i in range(0,len(classes)):
    categoria=np.where(datos[columnaObjetivo].values==classes[i])
    v[categoria[0],i]=1
    
#Dividir datos en entrenamineto y prueba
print('Dividiendo datos en entrenamiento y prueba...')
X_train, X_test, y_train, y_test = train_test_split(abs(datos[Col].values), v, test_size=porcentajeDataSetPruebas, stratify=datos[columnaObjetivo], random_state=semillaRandom)
# Imprimimos la cantidad de cada conjunto de datos.
print(f'Número de ejemplos en el conjunto de entrenamiento: {len(X_train)}')
print(f'Número de ejemplos en el conjunto de pruebas: {len(y_test)}')


# In[10]:


#Estructura de la red Neuronal
model = Sequential()
model.add(Dense(500, activation='relu', input_dim=10))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(5, activation='sigmoid'))
# Compilar modelo
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#Imprimir modelo
model.summary()


# In[11]:


# Entrenar Modelo
print('Entrenando Modelo...')
model.fit(X_train, y_train, epochs=150)


# In[12]:


#Evaluar Modelo
print('Evaluar Modelo')
modelo_rnn=model.evaluate(X_test, y_test, verbose=0)
print(f"La precisión del modelo Secuencial con el set de prueba es: {modelo_rnn[1]:.6f}")


# In[13]:


for i in range(0,len(classes)):
    categoria=np.where(y_test[:,i]==1)
    print('Evaluar Modelo')
    NN=model.evaluate(X_test[categoria], y_test[categoria], verbose=0)
    print(f"La precisión del modelo Secuencial con el set de prueba es: {NN[1]:.6f}")


# In[14]:


#Guardando Modelo
print('Guardando Modelo RNN')
json_model = model.to_json()
open(pathDeTrabajo+'RNN_Arquitactura.json', 'w').write(json_model)
model.save_weights(pathDeTrabajo+'RNN_modelo.h5')
#model = model_from_json(open(pathDeTrabajo+'RNN_Arquitactura.json').read())
#model.load_weights(pathDeTrabajo+'RNN_modelo.h5')
#model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#model.summary()


# In[15]:


#Gradient Bossting
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Extraemos el conjunto de clases.
classes = datos[columnaObjetivo].unique().tolist()

# Convertimos las clases (strings) a enteros, utilizando el índice que les corresponde en el dataframe.
datos[columnaObjetivo] = datos[columnaObjetivo].map(classes.index)

# Dividimos el data set
X_train, X_test, y_train, y_test = train_test_split(abs(datos[Col]), datos[columnaObjetivo], test_size=porcentajeDataSetPruebas, stratify=datos[columnaObjetivo], random_state=semillaRandom)

model_gbt_sk = GradientBoostingClassifier()
model_gbt_sk.fit(X_train, y_train)

acc_gbt_sk = accuracy_score(y_test.values, model_gbt_sk.predict(X_test))
print(f"La precisión del modelo Gradient Boosting Classifier para el set de prueba es: {acc_gbt_sk:.6f}")

#Guardar modelo sklear
print('Guardando modelo GradientBoosting')
archivo_joblib = pathDeTrabajo+"GradientBoosting.pkl"
joblib.dump(model_gbt_sk,archivo_joblib)


# In[16]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier
model_rf_sk = RandomForestClassifier()
model_rf_sk.fit(X_train, y_train)
acc_rf_sk = accuracy_score(y_test.values, model_rf_sk.predict(X_test))
print(f"La precisión del modelo Random Forest Classifier para el set de prueba es: {acc_rf_sk:.6f}")

#Guardar modelo sklear
print('Guardando modelo Random Forest')
archivo_joblib = pathDeTrabajo+"RandomForest.pkl"
joblib.dump(model_rf_sk,archivo_joblib)


# In[17]:


for i in range(0,len(classes)):
    categoria=np.where(y_test==i)
    print('Evaluar Modelo')
    Rf= accuracy_score(y_test.values[categoria], model_rf_sk.predict(X_test.values[categoria]))
    print(f"La precisión del modelo Random Forest Classifier para el set de prueba es: {Rf:.6f}")


# In[18]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier
model_dtc_sk = DecisionTreeClassifier()
model_dtc_sk.fit(X_train, y_train)
acc_dtc_sk = accuracy_score(y_test.values, model_dtc_sk.predict(X_test))
print(f"La precisión del modelo Decision Tree Classifier para el set de prueba es:{acc_dtc_sk:.6f}")


# In[19]:


# K Neighbors
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 6)
knn.fit(X_train, y_train)
print(f"La precisión del modelo K Neighbors Classifier para el set de prueba es:{knn.score(X_test, y_test):.6f}")


# In[ ]:





# In[ ]:





# In[20]:


from keras.models import model_from_json
model = model_from_json(open(pathDeTrabajo+'RNN_Arquitactura.json').read())
model.load_weights(pathDeTrabajo+'RNN_modelo.h5')
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()


# In[21]:


NN=model.predict(X_test[1:2])
print(NN)


# In[22]:


Posiciones=['a','be','ce','de','e','efe']
out=np.where(100*NN== np.amax(100*NN))
print(out)
print(Posiciones[int(out[1])])


# In[ ]:





# In[23]:


#Guardar modelo sklear
print('Guardando modelo Random Forest')
archivo_joblib = pathDeTrabajo+"RandomForest.pkl"
joblib.dump(model_rf_sk,archivo_joblib)


# In[24]:


eje = joblib.load(pathDeTrabajo+"RandomForest.pkl")
X_train, X_test, y_train, y_test = train_test_split(abs(datos[Col]), datos[columnaObjetivo], test_size=porcentajeDataSetPruebas, stratify=datos[columnaObjetivo], random_state=semillaRandom)
b=eje.predict(X_test[0:1])
print(b)


# In[25]:


eje.predict(X_test[0:1])


# In[ ]:


X_test.values


# In[ ]:


abs(X_test[0:1])


# In[ ]:


print(datos)


# In[ ]:




