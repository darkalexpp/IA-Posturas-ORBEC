#!/usr/bin/env python
# coding: utf-8

# En linux
# export NUITRACK_HOME="/usr/etc/nuitrack"
# export LD_LIBRARY_PATH={$LD_LIBRARY_PATH}:"/usr/local/lib/nuitrack"

# In[1]:


from PyNuitrack import py_nuitrack
import cv2
from datetime import datetime
from itertools import cycle
import numpy as np
import pandas as pd
import math
import tensorflow as tf
import joblib
import os
import time
import keras
from keras.models import model_from_json


# In[2]:

# Configuracion de camara
camaraNombre = 'Astra'
camaraNumeroSerie = 'AS3M32300AV'
# configurar colores cv2
modes = ["depth", "color"] #defecto del sdk en modo color, deshabilitar la cámara principal en conf windows
mode = modes[1]
tiempoDeEsperaPorCiclo = 33/1000 # en segundos 28fps
pathDeTrabajo='./'

Posiciones=['A','B','C','D','E','Desconocido']
Ang=pd.DataFrame(columns =['A1','A2','A3','A4','A5','A7','A8','A9','A10','A11'])
#metodo='RN' #Red Neuronal
categoriaActual='Desconocido'
letraAesASCII=97; 

#valores para el esqueleto
cuerpo = [1,2,3,4]
brazoDerecho = [5,6,7,8,9]
brazoIzquierdo = [10,11,12,13,14]
piernaDerecha = [4,18,19,20]
piernaIzquierda = [4,15,16,17]
esqueleto = [
            { "extr": cuerpo, "colorPunto":(164, 0, 0), "colorLine":(164, 0, 0)},
            { "extr": brazoDerecho, "colorPunto":(0, 150, 0), "colorLine":(0, 150, 0)},
            { "extr": brazoIzquierdo, "colorPunto":(0, 150, 0), "colorLine":(0, 150, 0)},
            { "extr": piernaDerecha, "colorPunto":(0, 0, 150), "colorLine":(0, 0, 150)},
            { "extr": piernaIzquierda, "colorPunto":(0, 0, 150), "colorLine":(0, 0, 150)},
            ]
#Valor de prediccion
out1 = [len(Posiciones)-1];
out2 = [len(Posiciones)-1];
out3 = [len(Posiciones)-1];


# In[3]:



if not os.path.exists(pathDeTrabajo):
    print('No existe el directorio de trabajo')
    exit

#Cargando modelo RN
if os.path.isfile(pathDeTrabajo+'RNN_Arquitactura.json'):  # Si existe
    model = model_from_json(open(pathDeTrabajo+'RNN_Arquitactura.json').read())
    model.load_weights(pathDeTrabajo+'RNN_modelo.h5')
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()
else:
    print('El archivo: '+pathDeTrabajo+'RNN_Arquitactura.json no existe')
    exit

#Cargando modelo RF
if os.path.isfile(pathDeTrabajo+'RandomForest.pkl'):  # Si existe
    modelRF = joblib.load(pathDeTrabajo+"RandomForest.pkl")
else:
    print('El archivo: '+pathDeTrabajo+'RandomForest.pkl no existe')
    exit

#Cargando modelo GB
if os.path.isfile(pathDeTrabajo+'GradientBoosting.pkl'):  # Si existe
    modelGB = joblib.load(pathDeTrabajo+"GradientBoosting.pkl")
else:
    print('El archivo: '+pathDeTrabajo+'GradientBoosting.pkl no existe')
    exit

if os.path.isfile(pathDeTrabajo + 'Datos.csv'):  # Si existe
    Dang = pd.read_csv(pathDeTrabajo + 'Datos.csv', sep=';', index_col=[0])
    if Dang.shape[0] >0:
        N = Dang.iloc[-1, 0] + 1
    else:
        N=1
else:  # No existe el archivo
    N = 1
    if not os.path.exists(pathDeTrabajo+'Captura'):
    	os.makedirs(pathDeTrabajo+'Captura')
    Dang = pd.DataFrame(
        columns=['N', 'Fecha', 'A1', 'A2', 'A3', 'A4', 'A5', 'A7', 'A8', 'A9', 'A10', 'A11',
                 'RN','RF','GB','Postura_correcta'])

# In[11]:


def draw_skeleton(image):
    global out1,out2,out3
    for skel in data.skeletons:
        for extremidadDibujo in esqueleto:
            for i in range(0,len(extremidadDibujo.get("extr"))):
                p1 = (round(skel[extremidadDibujo.get("extr")[i]].projection[0]), round(skel[extremidadDibujo.get("extr")[i]].projection[1]))
                #Graficar recuadro y texto de las predicciones
                if extremidadDibujo.get("extr")[i]==1:
                    cv2.rectangle(image, (10,10), (60,60),(0,0,0), 50)
                    cv2.putText(image,"RN: "+Posiciones[int(out1[0])], (1,20), cv2.FONT_ITALIC,0.7, (255,255,255), 2, bottomLeftOrigin = False)
                    cv2.putText(image,"RF: "+Posiciones[int(out2[0])], (1,45), cv2.FONT_ITALIC,0.7, (255,255,255), 2, bottomLeftOrigin = False)
                    cv2.putText(image,"GB: "+Posiciones[int(out3[0])], (1,70), cv2.FONT_ITALIC,0.7, (255,255,255), 2, bottomLeftOrigin = False)
                cv2.circle(image, p1, 5, extremidadDibujo.get("colorPunto"), -1)
                if i+1 < len(extremidadDibujo.get("extr")):
                    p2 = (round(skel[extremidadDibujo.get("extr")[i+1]].projection[0]), round(skel[extremidadDibujo.get("extr")[i+1]].projection[1]))
                    if p2 == (0,0):
                        continue
                    cv2.line(image,p1,p2,extremidadDibujo.get("colorLine"),2)

    #cv2.putText(image,categoriaActual, (10,10), cv2.FONT_HERSHEY_SIMPLEX, 25, (255,255,255), 10, bottomLeftOrigin = False)


# In[5]:

#Captura imagen de la camara y el esqueleto
def captura():
    nuitrack.update()
    global data
    data = nuitrack.get_skeleton()
    data_instance = nuitrack.get_instance()
    img_depth = nuitrack.get_depth_data()
    if img_depth.size:
        cv2.normalize(img_depth, img_depth, 0, 255, cv2.NORM_MINMAX)
        img_depth = np.array(cv2.cvtColor(img_depth, cv2.COLOR_GRAY2RGB), dtype=np.uint8)
        if mode == "depth":
            draw_skeleton(img_depth)
            cv2.imshow('Image', img_depth)
            return (img_depth)
        if mode == "color":
            img_color = nuitrack.get_color_data()
            draw_skeleton(img_color)
            if img_color.size:
                cv2.imshow('Image', img_color)
                return (img_color)


# In[6]:


def angulo(P1,P2): #[Punto a medir,Punto referencia]
    c=P1-P2;
    angle=math.atan2(c[1],c[0])
    return(math.degrees(angle))


# In[7]:


#Calculo los Angulos
def calculos():
    global Puntos
    global metodo
    for skeleton in data.skeletons: #Separo los puntos en varias variables
        #print(skeleton.user_id)
        head=skeleton.head   #print(skeleton.head)
        neck=skeleton.neck   #print(skeleton.neck)
        torso=skeleton.torso #print(skeleton.torso)
        waist=skeleton.waist #print(skeleton.waist)
        #lado izquierdo
        left_collar=skeleton.left_collar
        left_shoulder=skeleton.left_shoulder
        left_elbow=skeleton.left_elbow
        left_wrist=skeleton.left_wrist
        left_hand=skeleton.left_hand
        left_hip=skeleton.left_hip
        left_knee=skeleton.left_knee
        left_ankle=skeleton.left_ankle
        #lado izquierdo
        right_collar=skeleton.right_collar
        right_shoulder=skeleton.right_shoulder
        right_elbow=skeleton.right_elbow
        right_wrist=skeleton.right_wrist
        right_hand=skeleton.right_hand
        right_hip=skeleton.right_hip
        right_knee=skeleton.right_knee
        right_ankle=skeleton.right_ankle
        
        #Lateral
        #Angulos del torso
        A1=90-angulo(head.real[[0,1]],neck.real[[0,1]])      #cabeza-cuello
        A2=90-angulo(neck.real[[0,1]],torso.real[[0,1]])     #cuello-torso
        A3=90-angulo(torso.real[[0,1]],waist.real[[0,1]])    #torso-cintura
        #Angulo del brazo
        A4=angulo(right_shoulder.real[[0,1]],right_elbow.real[[0,1]])-angulo(right_wrist.real[[0,1]],right_elbow.real[[0,1]])    #Hombro-codo-muñeca
        #Angulo de la pierna
        A5=angulo(waist.real[[0,1]],right_hip.real[[0,1]])-angulo(right_knee.real[[0,1]],right_hip.real[[0,1]])    #Cintura-cadera-rodilla
        #Angulo de la rodilla
        #A6=angulo(right_hip.real[[0,1]],right_knee.real[[0,1]])-angulo(right_knee.real[[0,1]],right_ankle.real[[0,1]])    #Cadera-rodilla-tobillo
        
        #Frontal
        #Angulos del torso
        A7=90-angulo(head.real[[2,1]],torso.real[[2,1]])      #cabeza-torso
        A8=90-angulo(torso.real[[2,1]],waist.real[[2,1]])    #torso-cintura
        #Angulos del brazo
        A9=90-angulo(right_shoulder.real[[2,1]],right_elbow.real[[2,1]])    #Hombro-codo Derecha
        A10=90-angulo(left_shoulder.real[[2,1]],left_elbow.real[[2,1]])    #Hombro-codo Izquierda
        #Angulos de la pierna
        A11=360-abs(angulo(right_knee.real[[2,1]],waist.real[[2,1]]))+abs(angulo(left_knee.real[[2,1]],waist.real[[2,1]]))      #Cadera_derecha-Cintura-Cadera Izquierda
        #A12=90-angulo(right_knee.real[[2,1]],right_ankle.real[[2,1]])    #Rodilla-tobillo Derecha
        #A13=90-angulo(left_knee.real[[2,1]],left_ankle.real[[2,1]])    #Rodilla-tobillo Izquierda   
        return [A1, A2, A3, A4, A5, A7, A8,A9,A10, A11]
    return None
#///////////////////////////////////////////////////////  

# Predice usando los 3 modelos
def predecir(Angulos):
    global categoriaActual
    global Ang
    global out1,out2,out3
    
    Ang=pd.DataFrame([Angulos],columns=Ang.columns)
    # Red neuronal
    NN=model.predict(abs(Ang[0:1].values),verbose=0)
    out=np.where(100*NN== np.amax(100*NN))
    out1=out[1]
    # Random Forest
    out2=modelRF.predict(abs(Ang[0:1])) 
    # Gradient
    out3=modelGB.predict(abs(Ang[0:1])) 
                          

def guardar(categoria):
    global N
    global foto
    global muestra
    global Dang
    nombre = 'Imagen_' + str(N)
    if not os.path.exists(pathDeTrabajo+'clase_'+categoria):
        os.makedirs(pathDeTrabajo+'clase_'+categoria)
    cv2.imwrite(pathDeTrabajo+'clase_'+categoria+'/' + nombre + '.jpg', foto)
    [A1, A2, A3, A4, A5, A7, A8,A9,A10, A11]=muestra
    Dang= pd.concat([Dang, pd.DataFrame([[N, datetime.now(), A1, A2, A3, A4, A5, A7, A8,A9,A10, A11, Posiciones[int(out1[0])],Posiciones[int(out2[0])],Posiciones[int(out3[0])],categoria]],columns=Dang.columns, index=[nombre])])
    Dang.to_csv(pathDeTrabajo + 'Datos.csv', sep=';')
    N = N + 1

# In[8]:

# Inicializa la camara
nuitrack = py_nuitrack.Nuitrack()
nuitrack.init()
devices = nuitrack.get_device_list()
for dev in devices:
    if dev.get_name() == camaraNombre and camaraNumeroSerie == dev.get_serial_number():
        nuitrack.set_device(dev)
        print(dev.get_name(), dev.get_serial_number())
nuitrack.create_modules()
nuitrack.run()
nuitrack.set_config_value("Faces.ToUse", "true")#coordenada de la cabeza centro de la cabeza ---false->nariz
nuitrack.set_config_value("DepthProvider.Depth2ColorRegistration", "true")


# In[9]:

# Bucle principal
while 1:
    try:
        key = cv2.waitKey(1)

        if key == 27: #Presiona Esc para salir y guardar
            break
        
        for i in range(0,len(Posiciones)): # Recorre las categorias creadas
            if key == i+letraAesASCII:  # verifica si la tecla presionada corresponde al ASCII de la categoria
                guardar(Posiciones[i])

        foto=captura()
        muestra=calculos()
        if muestra is not None:
            predecir(muestra)

    except Exception as e:
        print('Ocurrio un error:')
        print(str(e))
        time.sleep(2000) #tiempo de espera en caso de falla
    finally:
        time.sleep(tiempoDeEsperaPorCiclo) #tiempo de espera en seg-> 28fps

cv2.destroyAllWindows()


# In[10]:


nuitrack.release()


# In[ ]:





# In[ ]:




