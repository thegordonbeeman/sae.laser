# Paramètres de la droite d'intersection des plans HOR et VER: y = ax + b
D_INT_A = 1.252
D_INT_B = 116

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