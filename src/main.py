import os, shutil

import cv2
import numpy as np

seq_dir = "C:\\Users\\Thomas Laburthe\\Documents\\Code\\sae.laser.images\\"
seq_name = "imgs2024-03-03_17_49_28.135995R"
seq_path = os.path.join(seq_dir, seq_name)
seq_zip = seq_path + ".zip"

if (os.path.isdir(seq_path)):
	print("La séquence d'images est un répertoire.")
elif (os.path.isfile(seq_zip)):
	print("La séquence d'images est une archive. Décompression...")

	try:
		shutil.unpack_archive(seq_zip, extract_dir=seq_dir)
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

with np.load(calib_path) as X:
	mtx, dist, rvecs, tvecs = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]

# Intrinsincs
alphau = mtx[0][0]
alphav = mtx[1][1]
pu = mtx[0][2]
pv = mtx[1][2]

print("alphau:", alphau)
print("alphav:", alphav)
print("pu:", pu)
print("pv:", pv)

Rcalib, _ = cv2.Rodrigues(rvecs[0])