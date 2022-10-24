#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
#from math import sqrt

#Parametros configurables
pathDeTrabajo='./'
porcentajeDataSetPruebas=0.2
semillaRandom=42

# In[2]:

#Leer datos
print('Importado datos del csv...')
datos = pd.read_csv(pathDeTrabajo+'Angulos.csv',sep=',', index_col=[0])

#Limpiar Datos
print('Limpiando Datos')
datos = datos[datos.Postura_correcta != ' ']
datos = datos[datos.A4 != '180']
datos = datos[datos.A9 != '0']
datos = datos[datos.A10 != '0']
datos = datos.round(3)
print('Tamanio del DataSet')
print(datos.shape)


# In[3]:


columnaObjetivo='Postura_correcta'
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

# In[4]:

#Estructura de la red Neuronal
model = Sequential()
model.add(Dense(500, activation='relu', input_dim=len(Col)))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(len(classes), activation='sigmoid'))
# Compilar modelo
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#Imprimir modelo
model.summary()


# In[5]:


# Entrenar Modelo
print('Entrenando Modelo...')
model.fit(X_train, y_train, epochs=250, verbose=0)


# In[7]:

#Evaluar Modelo
print('Evaluar Modelo')
modelo_rnn=model.evaluate(X_test, y_test, verbose=0)
print(f"La precisión del modelo Secuencial con el set de prueba es: {modelo_rnn[1]:.6f}")


# In[ ]:

#Guardando Modelo
print('Guardando Modelo RNN')
json_model = model.to_json()
open(pathDeTrabajo+'RNN_Arquitactura.json', 'w').write(json_model)
model.save_weights(pathDeTrabajo+'RNN_modelo.h5')
#model = model_from_json(open('E:\Capturas\model_architecture.json').read())
#model.load_weights('E:\Capturas\model.h5')
#model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#model.summary()


# In[13]:

#Gradient Bossting

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib

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

# In[9]:

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


# In[10]:

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
model_dtc_sk = DecisionTreeClassifier()
model_dtc_sk.fit(X_train, y_train)
acc_dtc_sk = accuracy_score(y_test.values, model_dtc_sk.predict(X_test))
print(f"La precisión del modelo Decision Tree Classifier para el set de prueba es:{acc_dtc_sk:.6f}")


# In[11]:

# K Neighbors
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 6)
knn.fit(X_train, y_train)
print(f"La precisión del modelo K Neighbors Classifier para el set de prueba es:{knn.score(X_test, y_test):.6f}")
