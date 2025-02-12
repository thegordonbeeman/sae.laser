import cv2
import numpy as np

class Camera():
	def __init__(self, calib_params):
		self.alphau = calib_params['mtx'][0][0]
		self.alphav = calib_params['mtx'][1][1]
		self.pu = calib_params['mtx'][0][2]
		self.pv = calib_params['mtx'][1][2]
		self.tvec = calib_params['tvecs'][0]
		self.rvec = calib_params['rvecs'][0]

		self.rot, _ = cv2.Rodrigues(self.rvec)

		self.update_iCc(noupdate=True)
		self.update_cRTw()

	def update_iCc(self, noupdate=False):
		self.iCc = np.float32([
			[self.alphau, 0, self.pu, 0],
			[0, self.alphav, self.pv, 0],
			[0, 0, 1, 0]
		])

		if (not noupdate):
			self.update_iCw()
	
	def update_cRTw(self, noupdate=False):
		self.cRTw = np.hstack((self.rot, self.tvec))
		self.cRTw = np.vstack((self.cRTw, [0, 0, 0, 1]))

		if (not noupdate):
			self.update_iCw()
	
	def update_iCw(self):
		self.iCw = self.iCc @ self.cRTw