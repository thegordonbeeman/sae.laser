import numpy as np

# Paramètres de la droite d'intersection des plans HOR et VER: y = ax + b
D_INT_A = 1.252
D_INT_B = 116

def gen_iCc(params):
	return np.float32([
		[params['alphau'], 0, params['pu'], 0],
		[0, params['alphav'], params['pv'], 0],
		[0, 0, 1, 0]])

def gen_cRTw(mRot, params):
	cRTw = np.hstack((mRot, params['tvec']))
	return np.vstack((cRTw, [0, 0, 0, 1]))

def dinter(x):
	return D_INT_A * x + D_INT_B

def dinter_i(x):
	return int(dinter(x))

# Le point 2D est il sur le plan vertical ?
def pt2_in_plver(px, py):
	return dinter(px) < py

# ~ horizontal ~
def pt2_in_plhor(px, py):
	return dinter(px) > py

# Projeter un point 2D sur un plan 3D
# 'plane' est un Vecteur 4
A = np.zeros((3, 3), np.float32)
B = np.zeros((3, 1), np.float32)
def pt2_to_pt3(pt2, plane, params):
	iCw = params['iCw']

	n = 0
	A[n][0]	= iCw[2][0]*pt2[0]-iCw[0][0]
	A[n][1]	= iCw[2][1]*pt2[0]-iCw[0][1]
	A[n][2]	= iCw[2][2]*pt2[0]-iCw[0][2]
	B[n]	=-iCw[2][3]*pt2[0]+iCw[0][3]

	n = 1
	A[n][0]	= iCw[2][0]*pt2[1]-iCw[1][0]
	A[n][1]	= iCw[2][1]*pt2[1]-iCw[1][1]
	A[n][2]	= iCw[2][2]*pt2[1]-iCw[1][2]
	B[n]	=-iCw[2][3]*pt2[1]+iCw[1][3]

	n = 2
	A[n][0]	= plane[0]
	A[n][1]	= plane[1]
	A[n][2]	= plane[2]
	B[n]	= plane[3]

	return np.linalg.inv(A) @ B