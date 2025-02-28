import os

import numpy as np
import matplotlib.pyplot as plt

# Nombre de points à afficher
SAMPLE_COUNT = 5000

array = np.load(os.path.join(os.getcwd(), "result.npy")

_is = np.random.choice(array.shape[1], SAMPLE_COUNT)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(array[0, _is], array[1, _is], array[2, _is])

ax.set_aspect('equal')
plt.show()