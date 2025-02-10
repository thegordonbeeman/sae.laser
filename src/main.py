import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os

np.set_printoptions(suppress=True)

def compareFloat(f1, f2):
    if f1 == 0 :
      return abs(f1-f2) < 1e-6
    return (abs((f1-f2))/f1)<1e-6

fileLocation = '/home/burei/Documents/IUT/robotique_s6/PHOTOS_SAE/sequences_imgs'
sequence = 'imgs2024-03-03_17_49_28.135995R'
fileName = fileLocation + '/' + sequence + '/' + 'CalibResult.npz'

pictureName = fileLocation + '/' + sequence +  '/' + 'im_00000R.png'

dimSquare=0.012
with np.load(fileName) as X:
  mtx, dist, rvecs, tvecs = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
  print("rvecs", rvecs)
  print("tvecs", tvecs)

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

iCc=np.float32([[alphau,0,pu,0],[0,alphav,pv,0],[0,0,1,0]])


def projeterPoint(M, iCw):
    #print("M:"+str(M))
    M = np.append(M, 1)
    #print("M:"+str(M))
    m=iCw@M

    mu=m[0]/m[2]
    mv=m[1]/m[2] 

    mu=int(round(mu,0))
    mv=int(round(mv,0))
    #print("mu:"+str(mu))
    #print("mv:"+str(mv))   
    return  (mu,mv)

iCw1 = None

for i in range(nbimages):
    
    matroti, jacobian=cv2.Rodrigues(rvecs[i]) #Rodrigues permet de calculer la matrice de rotation correspondante    
    axis = np.float32([[0,0,0], [3*dimSquare,0,0], [0,3*dimSquare,0], [0,0,3*dimSquare]]).reshape(-1,3)
    imgpts, jac = cv2.projectPoints(axis, rvecs[i], tvecs[i], mtx,None) #pour ne pas utiliser les distorsions
    cRTw=np.concatenate((matroti,tvecs[i]),axis=1)
    cRTw = np.vstack((cRTw,[0,0,0,1]))
    iCw=iCc@cRTw
    if i == 0:
      iCw1 = iCw
      print(iCw1.shape)

img = cv2.imread(pictureName, cv2.IMREAD_COLOR)
orig, X, Y, Z= projeterPoint((0,0,0), iCw1), projeterPoint((0.1,0,0), iCw1), projeterPoint((0,0.1,0), iCw1), projeterPoint((0,0,0.1), iCw1)

for x in range(5000):
  point = projeterPoint((-x*0.00001,0,0), iCw1)

   #print(point[0], 1.252 * point[0] + 116, point[1])
   
  if (1.252 * point[0] + 116 < point[1]):
      print(x)
      cv2.circle(img, point, 1, (255,0,0), 5)
      break

  
   
   #if compareFloat(1.252*point[0] + 116, point[1]):
      #print("intersection at :", point)

cv2.line(img, orig, X, (0,0,255), 3)
cv2.line(img, orig, Y, (0,255,0), 3)
cv2.line(img, orig, Z, (255,0,0), 3)
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
    