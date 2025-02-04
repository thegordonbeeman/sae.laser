import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os

print(os.getcwd())

fileLocation = 'sequences_imgs/'
sequence = 'imgs2024-03-03_17_49_28.135995R'
fileName = os.getcwd() + "/" + fileLocation + sequence + '/' + 'im_00000R.png'

img = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)

ret, t1 = cv2.threshold(img, 230,255, cv2.THRESH_BINARY)
redline = cv2.cvtColor(t1, cv2.IMREAD_COLOR)

contours, _ = cv2.findContours(t1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

'''print(len(contours))

approx = cv2.approxPolyDP(contours[0], 0.02, False)
print("appox", approx)
cv2.drawContours(redline, contours, 0, (0,0,255), 1)'''

edges = cv2.Canny(t1, 100, 200)

cdst = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cdstP = np.copy(cdst)

lines = cv2.HoughLines(edges, 1, np.pi / 180, 110, None, 0, 0)

print(lines)

if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        print(theta)
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

        cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)


cv2.line(redline, (0,0), (100,100), (0,0,255), 1)
cv2.imshow("image", img)
'''cv2.imshow("threshold", t1)
cv2.imshow("redline", redline)'''
cv2.imshow("canny", edges)
cv2.imshow("cdst", cdst)
cv2.waitKey(0)
cv2.destroyAllWindows()


