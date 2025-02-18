import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from mpl_toolkits.mplot3d import Axes3D

fileLocation = '/home/burei/Documents/IUT/robotique_s6/PHOTOS_SAE/sequences_imgs'
sequence = 'imgs2024-03-03_17_49_28.135995R'
fileName = fileLocation +'/' + sequence + '/' + 'im_00000R.png'

def is_point_in_vertical(pX, pY):
    return 1.252 * pX + 116 < pY

def is_point_in_horizontal(pX, pY):
    return 1.252 * pX + 116 > pY

def fit_line_least_squares(x, y):
    num_points = len(x)
    x_sum = sum(x)
    y_sum = sum(y)
    x2_sum = 0
    xy_sum = 0
    for xi in x:
        x2_sum += xi**2
    for xi, yi in zip(x, y):
        xy_sum += xi*yi
    
    a = (num_points * xy_sum - x_sum * y_sum) / (num_points * x2_sum - x_sum**2)
    b = (x2_sum * y_sum - x_sum * xy_sum) / (num_points * x2_sum - x_sum**2)
    return a, b

def determine_plan(pA, pB, pC):
    vecB = pB - pA
    vecC = pC - pA
    vecNormal = np.cross(vecB, vecC)
    return vecNormal

img = cv2.imread(fileName)
ret, t1 = cv2.threshold(img, 230,255, cv2.THRESH_BINARY)
mask = cv2.cvtColor(t1, cv2.COLOR_BGR2GRAY)

laser_points = np.where(mask == 255)
pointsX = laser_points[1]
pointsY = laser_points[0]


alpha_u = 686.3954829545311
alpha_v = 689.4310861162195
pu = 398.65885238460527
pv = 218.87245412041145

tvec = np.array([
    [-0.04335364],
    [-0.00516875],
    [0.17832831]]
)

rvec = np.array([
    [0.88312591],
    [0.03631245],
    [-0.1481258]]
)

matR1,jac1=cv2.Rodrigues(rvec)

iCc=np.float32([[alpha_u,0,pu,0],[0,alpha_v,pv,0],[0,0,1,0]])

cRT1w = np.hstack((matR1,tvec))
cRT1w = np.vstack((cRT1w,[0,0,0,1]))

iC1w=iCc@cRT1w

cv2.line(img, (222, 394), (0, 116), (0,0,255), 1) #Intersection pV, pH

A = np.zeros((3,3), np.float32)
B = np.zeros((3,1), np.float32)

points_pHOZ = []
points_pVER = []
for mu, mv in zip(pointsX, pointsY):
    if is_point_in_horizontal(mu, mv):
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

        X = np.linalg.inv(A)@B
        points_pHOZ.append(X)

    elif is_point_in_vertical(mu, mv):
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
        A[n][0]=1
        A[n][1]=0
        A[n][2]=0
        B[n]=-0.04155

        X = np.linalg.inv(A)@B
        points_pVER.append(X)

points_pHOZ = np.array(points_pHOZ)
points_pVER = np.array(points_pVER)
'''cv2.circle(img, pointV, 1, (255,0,0), 2)
cv2.circle(img, pointH, 1, (255,0,0), 2)

print("test Phorizontal", is_point_in_horizontal(pointH[0], pointH[1]))
print("test Pvertical", is_point_in_vertical(pointV[0], pointV[1]))
cv2.imshow("test ineq plans", img)
cv2.imshow("seuillage", t1)

cv2.waitKey(0)
cv2.destroyAllWindows()'''

distance_x = -0.4155

fig = plt.figure()
ax = plt.axes(projection='3d')

yVer = np.array(points_pVER[:,1])
zVer = np.array(points_pVER[:,2])
a_v,b_v = fit_line_least_squares(yVer, zVer)

xline_v = np.ones(shape=len(yVer))*-0.04155
yline_v = np.linspace(0, 0.1, len(yVer))
zline_v = np.linspace(0, 0.1, len(yVer))*a_v+b_v


xHoz = np.array(points_pHOZ[:,0])
yHoz = np.array(points_pHOZ[:,1])
a_h,b_h = fit_line_least_squares(xHoz, yHoz)

zline_h = np.zeros(shape=len(yVer))
xline_h = np.linspace(0, 0.1, len(yVer))
yline_h = np.linspace(0, 0.1, len(yVer))*a_h+b_h

ax.plot3D(xline_v, yline_v, zline_v)  #Plot la ligne passant par les points laser du plan horizontal
ax.plot3D(xline_h, yline_h, zline_h)  #Plot la ligne passant par les points laser du plan vertical

ax.scatter(points_pHOZ[:,0], points_pHOZ[:,1], points_pHOZ[:,2], marker ='o', color='blue')
ax.scatter(points_pVER[:,0], points_pVER[:,1], points_pVER[:,2], marker ='^', color='green')

#Calcul des 3 points pour le plan laser
P1 = np.array([distance_x, np.mean(points_pVER[:,1]), 0]) #Point sur le plan laser et l'intersection
P2 = np.array([distance_x, np.max(np.abs(points_pVER[:,1])), np.max(np.abs(points_pVER[:,2]))]) #Point sur le plan laser et le plan vertical
P3 = np.array([np.max(np.abs(points_pHOZ[:,0])), np.max(np.abs(points_pHOZ[:,1])), 0]) #Point sur le plan laser et le plan horizontal

vN = determine_plan(P1, P2, P3)
vN = vN / np.linalg.norm(vN)
point_laser = points_pHOZ[20]
dist2orig = np.linalg.norm(point_laser)
print(point_laser)
print(vN@point_laser)
print(dist2orig)
d = dist2orig-vN@point_laser
print(vN[0], vN[1], vN[2], d)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_ylim(-5,5)

plt.show()
