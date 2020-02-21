import processor2000
import fitter2000
from cv2 import cv2
import jigsaw
import os
import queue
import threading
import time
import numpy as np

datadir = r'C:\jigsaw\data\2000'


font = cv2.FONT_HERSHEY_SIMPLEX

class calibrator():
	def __init__(self):
		self.factor_r = self.factor_g = self.factor_b = 0

	def calibrate(self, video, seconds):
		cal_time_start = time.time()
		calibrated = False
		while not calibrated:
			check,img = video.read()
			d = time.time() - cal_time_start
			hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
			text = ''
			hok = hisogramok(hsv)
			if hok:
				cal_time_start = time.time()
				cv2.putText(img, 'Remove peice', (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
				cv2.imshow('Webcam view', img)
				cv2.waitKey(1)
			elif (d < seconds): # allow 3 seconds for alignment following histogram match confirmation
				cv2.putText(img, 'Calibrating '+str(seconds-int(d)), (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
				cv2.imshow('Webcam view', img)
				cv2.waitKey(1)
			else:
				check,img = video.read()
				b,g,r = cv2.split(img)
				pixels = (img.shape[0]*img.shape[1])
				avg_r =  r.sum() / pixels
				avg_g =  g.sum() / pixels
				avg_b =  b.sum() / pixels
				blur = cv2.GaussianBlur(img,(65,65),0)
				blur_b,blur_g,blur_r = cv2.split(blur)
				self.factor_r = blur_r / avg_r
				self.factor_g = blur_g / avg_g
				self.factor_b = blur_b / avg_b
				cv2.waitKey(1)
				calibrated = True

	def capture(self, video):
		check,img = video.read()
		b,g,r = cv2.split(img)
		r_ = r / self.factor_r
		g_ = g / self.factor_g
		b_ = b / self.factor_b

		r = np.uint8(np.clip(r_, 0, 255))
		g = np.uint8(np.clip(g_, 0, 255))
		b = np.uint8(np.clip(b_, 0, 255))

		return cv2.merge((b,g,r))


class database():
	foldernames = ['lightgreen','white','yellow','orange']
	sx = 10
	sy = 15
	
	def getNextId(self):
		while (os.path.exists(database.id2path(self.id))):
			self.id += 1
		return self.id

	@staticmethod
	def id2path(id):
		return datadir + '\\png' + '\\' + database.id2filename(id) + '.png'

	@staticmethod
	def alum(i):
		return chr(ord('a')+i)
		
	@staticmethod
	def getfilename(page,x,y):
		name =  database.foldernames[page] + '/' + database.alum(x) + '_' + str(y+1) 
		return name
	
	@staticmethod	
	def id2filename(id):
		page,x,y = database.id2crd(id)
		path = database.getfilename(page,x,y)
		return path

	@staticmethod	
	def id2shortname(id):
		page,x,y = database.id2crd(id)
		return database.alum(x) + '_' + str(1+y)

	@staticmethod
	def id2crd(id):
		per_page = database.sx*database.sy
		page = id // per_page
		y = (id % per_page) // database.sx
		x = (id % database.sx)
		return (page,x,y)
		
	def __init__(self):
		for f in database.foldernames:
			path = datadir + '\\png' + '\\' + f
			if (not os.path.isdir(path)):
				os.mkdir(path)
			path = datadir + '\\npz' + '\\' + f
			if (not os.path.isdir(path)):
				os.mkdir(path)
		self.id = 0
		self.getNextId()

def hisogramok(hsv):
	h,s,v = cv2.split(hsv)
	blacksPixels = cv2.inRange(s, 0, 128)
	tot =  blacksPixels.sum() / 255
	pct = tot / (h.shape[0]*h.shape[1])
	return (pct >= 0.25)

def wait4histogram(xvid, video, seconds, name=''):
	nohist = time.time()
	if (seconds > 0):
		while(True):
			img = xvid.capture(video)
			hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
			text = ''
			hok = hisogramok(hsv)
			d = time.time() - nohist
			if not (hok):
				nohist = time.time()
				text = 'Not detected'
			else:
				text = 'Capture in '+str(seconds-int(d))
			if (d >= seconds):
				cv2.putText(img, 'Analysing', (10,250), font, 3, (255, 0, 0), 2, cv2.LINE_AA)
				cv2.putText(img, name, (10,450), font, 3, (255, 0, 0), 2, cv2.LINE_AA)
				cv2.imshow('Webcam view', img)
				cv2.waitKey(1)
				return
			cv2.putText(img, text, (10,250), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
			cv2.imshow('Webcam view', img)
			cv2.waitKey(1)
	else:
		while(True):
			img = xvid.capture(video)
			hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
			text = ''
			hok = hisogramok(hsv)
			if not (hok): return
			cv2.putText(img, 'Captured', (10,250), font, 3, (255, 0, 0), 2, cv2.LINE_AA)
			cv2.putText(img, name, (10,450), font, 3, (255, 0, 0), 2, cv2.LINE_AA)
			cv2.imshow('Webcam view', img)
			cv2.waitKey(1)

def main():
	d = database()
	video = cv2.VideoCapture(0)

	# calibrate:
	xvid = calibrator()
	xvid.calibrate(video, 2)

	while True:
		id = d.getNextId()
		pngfile = database.id2path(id)
		shortname = database.id2shortname(id)
		wait4histogram(xvid,video, 3, shortname)
		img = xvid.capture(video)
		p = processor2000.process_cam(img)
		p.id = database.id2filename(id)
		#p.show(True)
		p.save()
		cv2.imwrite(pngfile, img)
		wait4histogram(xvid,video,-1, shortname)

	

main()