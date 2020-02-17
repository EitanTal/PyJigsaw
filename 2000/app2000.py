import processor2000
import fitter2000
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

def typeAgrees(t1, t2):
	if (t1 == 0) and (t2 == 0): return True
	if (t1 > 0) and (t2 > 0): return True
	if (t1 < 0) and (t2 < 0): return True
	return False

def findit(p, database):
	cutoffLenScore = 16
	cutoffAngScore = 5
	cutoffGauge = 15
	cutoffScore = 1984
	matches = []

	for rot in range(4):
		for q in database:
			p.orient(rot)
			geoscore = 0
			diffscore = 0
			for side in range(4):
				# sanity: check that types match up
				if not typeAgrees(p.sidetype[side], q.sidetype[side]): break
				# geo: check that side lens and angles roughly match up
				len_diff = abs(p.sidelen[side] - q.sidelen[side])
				ang_diff = abs(p.ang[side] - q.ang[side])
				if (len_diff > cutoffLenScore): break
				if (ang_diff > cutoffAngScore): break
				# gauge: check that gauge roughly matches up
				gauge_diff = abs(p.sidetype[side] - q.sidetype[side])
				if (gauge_diff > cutoffGauge): break
				# fitter analysis score...
				score = fitter2000.fitProfileToItself(p.profile[side],q.profile[side],p.sidetype[side],q.sidetype[side])
				if (score > cutoffScore): break
				geoscore += 1984
				diffscore += score
			else:
				match = [p.id, diffscore, geoscore]
				matches += [match]

	matches = sorted(matches,key=lambda x: x[1])
	if (len(matches) == 0): print ('-- No Matches --')
	for i,elem in enumerate(matches):
		print( i,': ', elem[0], '  \t(',elem[1],')', '  \t(',elem[2],elem[3],')' )

	cmdline = input(">").strip()
	if (cmdline.isnumeric()):
		return matches[int(cmdline)][0]
	else:
		return None

def saveAndRemove(x, img, database):
	cv2.imwrite(datadir + '\\' + 'cam' + '\\' + x + '.png', img)
	for j in database:
		if (j.id == x):
			database.remove(j)
			break

main()