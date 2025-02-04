import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os

fileLocation = 'sequences_imgs/'
sequence = 'imgs2024-03-03_17_49_28.135995R'
dirPath = os.getcwd() + "/" + fileLocation + sequence + '/'

files = os.listdir(dirPath)

files = [f for f in files if not ("Calib" in f or "calib" in f)]

files.remove("im_00001R.png")
files.remove("im_00002R.png")
files.remove("im_00000R.png")
files.sort()

dest = fileLocation + "thresholds_1/"

for f in files:
    filePath = os.path.join(dirPath, f)
    img = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)
    ret, t1 = cv2.threshold(img, 50,255, cv2.THRESH_BINARY)
    cv2.imwrite(dest+f, t1)
