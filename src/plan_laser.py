import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os

fileLocation = '/home/burei/Documents/IUT/robotique_s6/PHOTOS_SAE/sequences_imgs'
sequence = 'imgs2024-03-03_17_49_28.135995R'
fileName = fileLocation +'/' + sequence + '/' + 'im_00000R.png'

def is_point_in_vertical(pX, pY):
    return 1.252 * pX + 116 < pY

def is_point_in_horizontal(pX, pY):
    return 1.252 * pX + 116 > pY

def fit_line_least_squares(datapoints):
    n = len(datapoints)
    m = 0

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
pointV = (75, 396)
pointH = (197, 102)

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
    print(X)
'''cv2.circle(img, pointV, 1, (255,0,0), 2)
cv2.circle(img, pointH, 1, (255,0,0), 2)

print("test Phorizontal", is_point_in_horizontal(pointH[0], pointH[1]))
print("test Pvertical", is_point_in_vertical(pointV[0], pointV[1]))
cv2.imshow("test ineq plans", img)
cv2.imshow("seuillage", t1)

cv2.waitKey(0)
cv2.destroyAllWindows()'''


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
xs = [p[0] for p in points_pVER]
ys = [p[1] for p in points_pVER]
zs = [p[2] for p in points_pVER]
ax.scatter(xs, ys, zs, marker ='^', color='green')

xs = [p[0] for p in points_pHOZ]
ys = [p[1] for p in points_pHOZ]
zs = [p[2] for p in points_pHOZ]
ax.scatter(xs, ys, zs, marker ='o', color='blue')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_ylim(-5,5)

plt.show()
