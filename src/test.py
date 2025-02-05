import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os

print(os.getcwd())

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

edges = cv2.Canny(t1, 100, 200)
edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
cv2.line(edges, (222, 394), (0, 116), (0,0,255), 1)
cv2.line(img, (222, 394), (0, 116), (0,0,255), 1)

cv2.imshow("image", img)
'''cv2.imshow("threshold", t1)
cv2.imshow("redline", redline)'''
cv2.imshow("canny", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()


