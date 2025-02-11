import os, time, shutil, glob

import cv2
import numpy as np
import matplotlib.pyplot as plt

import utils as u

DEBUG = False
DEBUG_MAX_FRAMES = 100

SEQ_DIR = "C:\\Users\\Thomas Laburthe\\Documents\\Code\\sae.laser.images\\"
SEQ_NAME = "imgs2024-03-03_17_49_28.135995R"

PLANE_VER = np.array([0, 0, 1, 0])
PLANE_HOR = np.array([1, 0, 0,-0.04155])

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

with np.load(calib_path) as X:
	mtx, dist, rvecs, tvecs = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]

# Intrinsincs
params = {
	'alphau' : mtx[0][0],
	'alphav' : mtx[1][1],
	'pu' : mtx[0][2],
	'pv' : mtx[1][2],
	'tvec' : tvecs[0],
	'rvec' : rvecs[0],
}

if (DEBUG):
	print(params)

Rcalib, _ = cv2.Rodrigues(params['rvec'])

params['iCc'] = u.gen_iCc(params)
params['cRTw'] = u.gen_cRTw(Rcalib, params)

params['iCw'] = params['iCc'] @ params['cRTw']

frames_paths = glob.glob(os.path.join(seq_path, "im_*R.png"))[3:]

frame_count = min(DEBUG_MAX_FRAMES, len(frames_paths))

tim_rd = np.empty((frame_count))
tim_th = np.empty((frame_count))
tim_pl = np.empty((frame_count))

for frame_index, frame_path in enumerate(frames_paths):
	if (True and frame_index + 1 > frame_count):
		break

	start_time = time.time()

	frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
	fsize = frame.shape

	tim_rd[frame_index] = time.time() - start_time
	start_time = time.time()

	# Histogramme
	if False:
		hist = cv2.calcHist([frame], [0], None, [256], [0, 256])
		plt.plot(hist, color='b')
		plt.show()

	# Traitement de la frame
	_, th1 = cv2.threshold(frame, 35, 255, cv2.THRESH_BINARY)

	tim_th[frame_index] = time.time() - start_time
	start_time = time.time()

	# Liste des coordonnées des points 2D du laser
	pt2s = np.where(th1 != 0)
	pt2s_plver, pt2s_plhor = [], []
	pt3s_plver, pt3s_plhor = [], []
	for x, y in zip(pt2s[1], pt2s[0]):
		if (th1[y, x] != 0):
			if (u.pt2_in_plver(x, y)):
				pt2s_plver.append((x, y))
				pt3s_plver.append(u.pt2_to_pt3((x, y), PLANE_VER, params))
			else:
				pt2s_plhor.append((x, y))
				pt3s_plver.append(u.pt2_to_pt3((x, y), PLANE_HOR, params))

	tim_pl[frame_index] = time.time() - start_time
	start_time = time.time()

	# Mode debug pour voir les calculs
	if (not DEBUG):
		continue

	frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

	cv2.line(frame, (0, u.dinter_i(0)), (fsize[1], u.dinter_i(fsize[1])), (0,0,255), 1)

	for pt2 in pt2s_plhor:
		frame[pt2[1], pt2[0]] = (0,255,0)
	
	for pt2 in pt2s_plver:
		frame[pt2[1], pt2[0]] = (0,0,255)

	cv2.imshow("Previsualisation", frame)
	cv2.waitKey(0)

print("Temps de lecture moyen:", np.mean(tim_rd))
print("Temps de treshold planaire moyen:", np.mean(tim_th))
print("Temps de triage planaire moyen:", np.mean(tim_pl))

if (DEBUG):
	cv2.destroyAllWindows()