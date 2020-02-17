import processor2000
from cv2 import cv2
import jigsaw
import os

datadir = r'C:\jigsaw\data\2000'

def main():
	database = getdb()
	while True:
		wait4histogram()
		delay()
		img = capture()
		p = processor2000.process_cam(img)
		x = findit(p, database)
		if (x): saveAndRemove(x, img, database)
		break

def getdb():
	fnames = os.listdir(datadir + '\\' + 'npz_tst')
	solved = os.listdir(datadir + '\\' + 'cam')
	npz = []
	for a in fnames:
		if a.endswith('.npz'): 
			id = a[:-4]
			png = id + '.png'
			if png in solved: continue
			x = jigsaw.jigsaw.load(id)
			npz += [x]
	print ('Loaded',len(npz),'peices from Database.')
	return npz

def wait4histogram():
	return 0

def delay():
	return 0

def capture():
	filename = r"C:\jigsaw\2000\rgb.png"
	return cv2.imread(filename)

def findit(p, database):
	return '2_28'

def saveAndRemove(x, img, database):
	cv2.imwrite(datadir + '\\' + 'cam' + '\\' + x + '.png', img)
	for j in database:
		if (j.id == x):
			database.remove(j)
			break

main()