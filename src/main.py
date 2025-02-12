import os, time, shutil, glob

import cv2
import numpy as np
import matplotlib.pyplot as plt

from camera import Camera
from geo import Point2D, Point3D, Plane
import utils as u

DEBUG = True
DEBUG_MAX_FRAMES = 20

SEQ_DIR = "C:\\Users\\Thomas Laburthe\\Documents\\Code\\sae.laser.images\\"
SEQ_NAME = "imgs2024-03-03_17_49_28.135995R"

PLANE_VER = Plane(0, 0, 1, 0)
PLANE_HOR = Plane(1, 0, 0,-0.04155)

seq_path = os.path.join(SEQ_DIR, SEQ_NAME)
seq_zip = seq_path + ".zip"

if (os.path.isdir(seq_path)):
	print("La séquence d'images est un répertoire.")
elif (os.path.isfile(seq_zip)):
	print("La séquence d'images est une archive. Décompression...")

	try:
		shutil.unpack_archive(seq_zip, extract_dir=SEQ_DIR)
		print("Décompression réussie!")	
	except:
		print("Impossible de décompresser la séquence d'images. Sortie.")
		exit()
else:
	print("La séquence d'images est introuvable. (Ni répertoire, ni archive)")
	exit()

calib_path = os.path.join(seq_path, "CalibResult.npz")

if (not os.path.isfile(calib_path)):
	print("Impossible de trouver le rapport de calibration:")
	print(calib_path)

calib_params = np.load(calib_path)
camera = Camera(calib_params)

frames_paths = glob.glob(os.path.join(seq_path, "im_*R.png"))[3:]
frame_count = min(DEBUG_MAX_FRAMES, len(frames_paths))

for frame_index, frame_path in enumerate(frames_paths):
	if (True and frame_index + 1 > frame_count):
		break

	frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
	fsize = frame.shape

	# Histogramme
	if False:
		hist = cv2.calcHist([frame], [0], None, [256], [0, 256])
		plt.plot(hist, color='b')
		plt.show()

	# Traitement de la frame
	_, th1 = cv2.threshold(frame, 35, 255, cv2.THRESH_BINARY)

	# points 2D et 3D du laser
	pt2s = np.where(th1 != 0)
	pt2s_plver, pt2s_plhor = [], []
	pt3s_plver, pt3s_plhor = [], []
	for x, y in zip(pt2s[1], pt2s[0]):
		if (th1[y, x] != 0):
			pt = Point2D(x, y)
			if (u.pt2_in_plver(x, y)):
				pt2s_plver.append(pt)
				pt3s_plver.append(pt.to3D(PLANE_VER, camera))
			else:
				pt2s_plhor.append(pt)
				pt3s_plhor.append(pt.to3D(PLANE_HOR, camera))

	# Mode debug pour voir les calculs
	if (not DEBUG):
		continue

	frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

	cv2.line(frame, (0, u.dinter_i(0)), (fsize[1], u.dinter_i(fsize[1])), (0,0,255), 1)

	for pt in pt2s_plhor:
		frame[pt.y, pt.x] = (0,255,0)
	
	for pt in pt2s_plver:
		frame[pt.y, pt.x] = (0,0,255)

	cv2.imshow("Previsualisation", frame)
	cv2.waitKey(0)

if (DEBUG):
	cv2.destroyAllWindows()