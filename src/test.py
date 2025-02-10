import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os

def ransac_line(samples, threshold, iterations = 500):
    np.random.seed(0)

    final_inliers = []
    final_outliers = []
    best_line = []

    score_sum = 0

    idx = np.random.randint(0, len(samples), size=2)
    while idx[0] == idx[1]:
        idx = np.random.randint(0, len(samples), size=2)

    pointsSegX = samples[idx[0]][0], samples[idx[1]][0]
    pointsSegY = samples[idx[0]][1], samples[idx[1]][1]

    x1=float(pointsSegX[0])
    y1=float(pointsSegY[0])
    x2=float(pointsSegX[1])
    y2=float(pointsSegY[1])

    A = np.array([[x1,y1], [x2,y2]], dtype=np.float64)
    B = np.array([[-1],[-1]])

    for p in samples:
        x = p[0]
        y = p[1]
        #dst = 

fileLocation = '/home/burei/Documents/IUT/robotique_s6/PHOTOS_SAE/sequences_imgs'
sequence = 'imgs2024-03-03_17_49_28.135995R'
fileName = fileLocation +'/' + sequence + '/' + 'im_00000R.png'
print(fileName)

img = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)

ret, t1 = cv2.threshold(img, 230,255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(t1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

'''print(len(contours))
approx = cv2.approxPolyDP(contours[0], 0.02, False)
print("appox", approx)
cv2.drawContours(redline, contours, 0, (0,0,255), 1)'''
print(len(contours))
edges = cv2.Canny(t1, 100, 200)
dst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
print(type(contours[0]))
ransac_line(contours[0], 0)

cv2.line(edges, (222, 394), (0, 116), (0,0,255), 1)
cv2.line(img, (222, 394), (0, 116), (0,0,255), 1)
print("edges", edges)
cv2.drawContours(edges, contours, 1, (255,0,0), 1)


cv2.imshow("image", img)
'''cv2.imshow("threshold", t1)
cv2.imshow("redline", redline)'''
cv2.imshow("canny", edges)
cv2.imshow("hough lines", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
