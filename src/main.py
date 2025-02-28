import os, time, shutil, glob

import cv2
import numpy as np
import matplotlib.pyplot as plt

from camera import Camera
from geo import Point2D, Point3D, Plane, to3D, pts_to_nvec, normalize
import utils as u

# Mode débug: montrera plus de détails sur le traitement de chaque image, inutilisable en production
# ATTENTION: DEBUG_MIN/MAX_FRAMES est valable même sans le mode débug
DEBUG = False
DEBUG_MIN_FRAMES = 100
DEBUG_MAX_FRAMES = 410

# DIR: Dossier ou se trouvent les séquences, NAME: Nom de la séquence à traiter
# La séquence peut être un répertoire ou une archive (ne pas aposer de .zip s'il s'agit d'une archive)
SEQ_DIR = "C:\\Users\\Thomas Laburthe\\Documents\\Code\\sae.laser.images\\"
SEQ_NAME = "imgs2024-03-03_17_49_28.135995R"

# Equations du plan horizontal et vertical
PLANE_HOR = np.array([0, 0, 1, 0])
PLANE_VER = np.array([1, 0, 0,-0.04155])

# Bounds 2D décrivant un carré englobant précisément l'objet à scanner
XMIN, XMAX = 300, 130
YMIN, YMAX = 560, 350

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

pt3s_final = np.zeros((3,1), np.float32)
valid_pts_cpt = 0

for frame_index, frame_path in enumerate(frames_paths):
	if (True and frame_index + 1 > frame_count):
		break

	if (frame_index < DEBUG_MIN_FRAMES):
		continue

	print(f"Image n°{frame_index}")

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
	pt2s = np.where(th1 == 255)
	pt_count = np.array(pt2s).shape[1]
	pt2s_plver, pt2s_plhor = np.empty((2, pt_count)), np.empty((2, pt_count))
	pt3s_plver, pt3s_plhor = np.empty((3, pt_count)), np.empty((3, pt_count))
	ih, iv = 0, 0
	for x, y in zip(pt2s[1], pt2s[0]):
		pt = np.array([x, y])
		if (u.pt2_in_plhor(x, y)):
			pt2s_plhor[:, ih] = pt[:]
			pt3s_plhor[:, ih] = to3D(pt, PLANE_HOR, camera)[:]
			ih += 1
		else:
			pt2s_plver[:, iv] = pt[:]
			pt3s_plver[:, iv] = to3D(pt, PLANE_VER, camera)[:]
			iv += 1
	
	pt2s_plhor = pt2s_plhor[:, :ih-1]
	pt3s_plhor = pt3s_plhor[:, :ih-1]
	pt2s_plver = pt2s_plver[:, :iv-1]
	pt3s_plver = pt3s_plver[:, :iv-1]

	P1 = np.array([PLANE_VER[3], np.min(pt3s_plver[1,:]), np.min(pt3s_plver[2,:])])
	P2 = np.array([PLANE_VER[3], np.max(pt3s_plver[1,:]), np.max(pt3s_plver[2,:])])
	P3 = np.array([np.max(pt3s_plhor[0,:]), np.max(pt3s_plhor[1,:]), 0])

	nvec = pts_to_nvec(P1, P2, P3)
	nvec = normalize(nvec)
	d = np.sum(nvec @ pt3s_plhor[:, :] / pt3s_plhor.shape[1])

	plaser = np.array([nvec[0], nvec[1], nvec[2], d], np.float32)

	pt3s_plaser = np.empty(pt3s_plhor.shape)
	_i = 0
	for i in range(pt2s_plhor.shape[1]):
		if (u.pt2_in_bounds(pt2s_plhor[0, i], pt2s_plhor[1, i], XMIN, XMAX, YMIN, YMAX)):
			pt3s_plaser[:, _i] = to3D(pt2s_plhor[:, i], plaser, camera)
			_i += 1
	
	pt3s_plaser = pt3s_plaser[:, :_i-1]
	pt3s_final = np.concatenate((pt3s_final, pt3s_plaser), axis=1)
	# pt3s_final[:, valid_pts_cpt:valid_pts_cpt + pt3s_plaser.shape[1]] = pt3s_plaser[:, :]
	valid_pts_cpt += pt3s_plaser.shape[1]

	# Mode debug pour voir les calculs
	if (not DEBUG):
		continue

	frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

	cv2.line(frame, (0, u.dinter_i(0)), (fsize[1], u.dinter_i(fsize[1])), (255,0,0), 2)

	frame[pt2s_plhor[1,:].astype(np.int32), pt2s_plhor[0,:].astype(np.int32)] = (0,0,255)
	frame[pt2s_plver[1,:].astype(np.int32), pt2s_plver[0,:].astype(np.int32)] = (0,255,0)

	fig = plt.figure()
	ax = fig.add_subplot(2, 1, 1)
	ax.imshow(frame)

	ax = fig.add_subplot(2, 1, 2, projection='3d')
	ax.scatter(P1[0], P1[1], P1[2], color='r')
	ax.scatter(P2[0], P2[1], P2[2], color='g')
	ax.scatter(P3[0], P3[1], P3[2], color='b')
	# ax.scatter(pt3s_plaser[0,:], pt3s_plaser[1,:], pt3s_plaser[2,:], color='g')
	# seli = np.random.choice(pt3s_plhor.shape[1], 25)
	# ax.scatter(pt3s_plhor[0,seli], pt3s_plhor[1,seli], pt3s_plhor[2,seli], color='g')
	# seli = np.random.choice(pt3s_plver.shape[1], 25)
	# ax.scatter(pt3s_plver[0,seli], pt3s_plver[1,seli], pt3s_plver[2,seli], color='r')
	ax.set_aspect('equal')

	plt.show()

pt3s_final = pt3s_final[:, :valid_pts_cpt-1]
seli = np.random.choice(pt3s_final.shape[1], 4000)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# plt.plot(range(pt3s_final.shape[1]), pt3s_final[0, :], 'r')
# plt.plot(range(pt3s_final.shape[1]), pt3s_final[1, :], 'g')
# plt.plot(range(pt3s_final.shape[1]), pt3s_final[2, :], 'b')
ax.scatter(pt3s_final[0, seli], pt3s_final[1, seli], pt3s_final[2, seli])
ax.set_aspect('equal')
plt.show()

np.save(os.path.join(os.getcwd(), "result"), pt3s_final[:, :])

if (DEBUG):
	cv2.destroyAllWindows()