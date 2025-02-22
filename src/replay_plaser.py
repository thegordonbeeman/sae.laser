import os, glob

import matplotlib.pyplot as plt
import matplotlib.animation as anm
import numpy as np

REPLAYS_DIR = "C:\\Users\\Thomas Laburthe\\Documents\\Code\\sae.laser\\src\\replays\\"

if (not os.path.isdir(REPLAYS_DIR)):
	print(f"Impossible de trouver le répertoire des replays...")
	exit()

replays_list = glob.glob(os.path.join(REPLAYS_DIR, "*"))
latest_replay = max(replays_list, key=os.path.getctime)

arrays = np.load(latest_replay)

replay_phoriz = arrays['phoriz']
replay_pverti = arrays['pverti']
replay_plaser = arrays['plaser']

replay_pt3s_phoriz = arrays['pt3s_phoriz']

print(replay_pt3s_phoriz.shape)

def surfpoints(plane):
	xx, yy = np.meshgrid(range(10), range(10))
	zz = np.copy(xx)

	a, b, c, d = tuple(plane)
	
	if (a != 0):
		return ((-b * yy - c * zz - d) * 1. / a, yy, zz)
	elif (b != 0):
		return (xx, (-a * xx - c * zz - d) * 1. / b, zz)
	else:
		return (xx, yy, (-a * xx - b * yy - d) * 1. / c)

# PLOT
fig = plt.figure()

ax = fig.add_subplot(projection='3d')

plane = replay_plaser[:, 0]
ax.plot_surface(*surfpoints(replay_phoriz), color='r', zorder=0)
ax.plot_surface(*surfpoints(replay_pverti), color='g', zorder=0)
ax.scatter(replay_pt3s_phoriz[0, :, 0], replay_pt3s_phoriz[1, :, 0], replay_pt3s_phoriz[2, :, 0], zorder=0.3)

ax.axes.set_xlim3d(left=0, right=0.2)
ax.axes.set_ylim3d(bottom=0, top=0.2)
ax.axes.set_zlim3d(bottom=0, top=0.2)

plt.show()