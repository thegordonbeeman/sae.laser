import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os


print(os.getcwd())

fileLocation = 'sequences_imgs/'
sequence = 'imgs2024-03-03_17_49_28.135995R'
fileName = os.getcwd() + "/" + fileLocation + sequence + '/' + 'CalibResult.npz'

dimSquare=0.012
with np.load(fileName) as X:
  mtx, dist, rvecs, tvecs = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

nbimages=len(rvecs)
alphau=mtx[0][0]
alphav=mtx[1][1]
pu=mtx[0][2]
pv=mtx[1][2]
print(
       "alpha u", alphau, "\n"
       "alpha v", alphav, "\n",
       "pu", pu, "\n",
       "pv", pv, "\n"
    )
 
for i in range(nbimages):  
    matroti, jacobian=cv2.Rodrigues(rvecs[i]) #Rodrigues permet de calculer la matrice de rotation correspondante    
    axis = np.float32([[0,0,0], [3*dimSquare,0,0], [0,3*dimSquare,0], [0,0,3*dimSquare]]).reshape(-1,3)
    imgpts, jac = cv2.projectPoints(axis, rvecs[i], tvecs[i], mtx,None) #pour ne pas utiliser les distorsions
    cRTw=np.concatenate((matroti,tvecs[i]),axis=1)
    cRTw = np.vstack((cRTw,[0,0,0,1]))
    print("cRTw: "+str(cRTw))

    