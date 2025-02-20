import numpy as np

# Matrices de calcul
A = np.zeros((3, 3), np.float32)
B = np.zeros((3, 1), np.float32)

def to3D(pt, plane, camera):
	iCw = camera.iCw

	n = 0
	A[n][0]	= iCw[2][0]*pt[0]-iCw[0][0]
	A[n][1]	= iCw[2][1]*pt[0]-iCw[0][1]
	A[n][2]	= iCw[2][2]*pt[0]-iCw[0][2]
	B[n]	=-iCw[2][3]*pt[0]+iCw[0][3]

	n = 1
	A[n][0]	= iCw[2][0]*pt[1]-iCw[1][0]
	A[n][1]	= iCw[2][1]*pt[1]-iCw[1][1]
	A[n][2]	= iCw[2][2]*pt[1]-iCw[1][2]
	B[n]	=-iCw[2][3]*pt[1]+iCw[1][3]

	n = 2
	A[n][0]	= plane[0]
	A[n][1]	= plane[1]
	A[n][2]	= plane[2]
	B[n]	= plane[3]

	X = np.linalg.inv(A) @ B

	return np.array(np.reshape(X, (3,)))

def pts_to_plane(pt1, pt2, pt3, off):
	nvec = normalize(pts_to_nvec(pt1, pt2, pt3))
	return np.array([nvec[0], nvec[1], nvec[2], off])

def pts_to_nvec(pt1, pt2, pt3):
	return np.cross(pt2 - pt1, pt3 - pt1)

def normalize(vec):
	return vec / np.linalg.norm(vec)

class Point2D():
	def __init__(self, x, y):
		self.x = x
		self.y = y
	
	def __iter__(self):
		return iter((self.x, self.y))
	
	def __array__(self):
		return np.array(list(self))
	
	def to3D(self, plane, camera):
		iCw = camera.iCw

		n = 0
		A[n][0]	= iCw[2][0]*self.x-iCw[0][0]
		A[n][1]	= iCw[2][1]*self.x-iCw[0][1]
		A[n][2]	= iCw[2][2]*self.x-iCw[0][2]
		B[n]	=-iCw[2][3]*self.x+iCw[0][3]

		n = 1
		A[n][0]	= iCw[2][0]*self.y-iCw[1][0]
		A[n][1]	= iCw[2][1]*self.y-iCw[1][1]
		A[n][2]	= iCw[2][2]*self.y-iCw[1][2]
		B[n]	=-iCw[2][3]*self.y+iCw[1][3]

		n = 2
		A[n][0]	= plane.a
		A[n][1]	= plane.b
		A[n][2]	= plane.c
		B[n]	= plane.d

		X = np.linalg.inv(A) @ B

		return Point3D(X[0], X[1], X[2])

class Point3D():
	def __init__(self, x, y, z):
		self.x = x
		self.y = y
		self.z = z

	def __iter__(self):
		return iter((self.x, self.y, self.z))
	
	def __array__(self):
		return np.array(list(self))

class Plane():
	def __init__(self, a, b, c, d):
		self.a = a
		self.b = b
		self.c = c
		self.d = d

	def __iter__(self):
		return iter((self.a, self.b, self.c, self.d))
	
	def __array__(self):
		return np.array(list(self))