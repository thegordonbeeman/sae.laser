import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os

fileLocation = '/home/burei/Documents/IUT/robotique_s6/PHOTOS_SAE/sequences_imgs'
sequence = 'imgs2024-03-03_17_49_28.135995R'
fileName = fileLocation +'/' + sequence + '/' + 'im_00000R.png'

img = cv2.imread(fileName)
ret, t1 = cv2.threshold(img, 230,255, cv2.THRESH_BINARY)

alpha_u = 686.3954829545311
alpha_v = 689.4310861162195
pu = 398.65885238460527
pv = 218.87245412041145

def is_point_in_vertical(pX, pY):
    return 1.252 * pX + 116 < pY

def is_point_in_horizontal(pX, pY):
    return 1.252 * pX + 116 > pY

cv2.line(img, (222, 394), (0, 116), (0,0,255), 1) #Intersection pV, pH
pointV = (75, 396)
pointH = (197, 102)

A = np.zeros((3,3), np.float32)
B = np.zeros((3,1), np.float32)

n=0
A[n][0]=iC1w[2][0]*mu-iC1w[0][0]
A[n][1]=iC1w[2][1]*mu-iC1w[0][1]
A[n][2]=iC1w[2][2]*mu-iC1w[0][2]
B[n]=  -iC1w[2][3]*mu+iC1w[0][3]

n=1
A[n][0]=iC1w[2][0]*mv-iC1w[1][0]
A[n][1]=iC1w[2][1]*mv-iC1w[1][1]
A[n][2]=iC1w[2][2]*mv-iC1w[1][2]
B[n]=  -iC1w[2][3]*mv+iC1w[1][3]

n=2
A[n][0]=0
A[n][1]=0
A[n][2]=1
B[n]=  0

X=np.linalg.inv(A)@B

cv2.circle(img, pointV, 1, (255,0,0), 2)
cv2.circle(img, pointH, 1, (255,0,0), 2)

print("test Phorizontal", is_point_in_horizontal(pointH[0], pointH[1]))
print("test Pvertical", is_point_in_vertical(pointV[0], pointV[1]))
cv2.imshow("test ineq plans", img)
cv2.imshow("seuillage", t1)

cv2.waitKey(0)
cv2.destroyAllWindows()
