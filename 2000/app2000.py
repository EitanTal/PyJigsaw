import processor2000
import fitter2000
from cv2 import cv2
import jigsaw
import os

datadir = r'C:\jigsaw\data\2000'
#sizecomp = 3.5  # shrink boxart sizes by 3.5%

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
	fnames = os.listdir(datadir + '\\' + 'npz')
	#fnames = os.listdir(datadir + '\\' + 'npz_tst_1')
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
	#filename = r"C:\jigsaw\2000\rgb.png"
	filename = r"C:\jigsaw\data\2000\t2.jpg"
	return cv2.imread(filename)

def typeAgrees(t1, t2):
	if (t1 == 0) and (t2 == 0): return True
	if (t1 > 0) and (t2 > 0): return True
	if (t1 < 0) and (t2 < 0): return True
	return False

def findit(p, database):
	cutoffLenScore = 64
	cutoffAngScore = 10
	cutoffGauge = 60
	cutoffGeoScore = 1984

	# fast GEO match:
	geomatch = []
	for q in database:
		for rot in range(4):
			p.orient(rot)
			geoscore = 0
			for side in range(4):
				camlen = p.sidelen[side] * 1.035
				# sanity: check that types match up
				if not typeAgrees(p.sidetype[side], q.sidetype[side]): break
				# geo: check that side lens and angles roughly match up
				len_diff = abs(camlen - q.sidelen[side])
				ang_diff = abs(p.ang[side] - q.ang[side])
				if (len_diff > cutoffLenScore): break
				if (ang_diff > cutoffAngScore): break
				# gauge: check that gauge roughly matches up
				gauge_diff = abs(p.sidetype[side] - q.sidetype[side])
				geoscore += ang_diff + len_diff
				if (gauge_diff > cutoffGauge): break
			else:
				if (geoscore > cutoffGeoScore): continue
				match = [q.id, geoscore, rot]
				geomatch += [match]

	# slow FITTER match:
	matches = []
	cutoffScore = 5000
	for qid,geoscore,rot in geomatch:
		for q in database:
			if (qid != q.id): continue
			diffscore = 0
			p.orient(rot)
			for side in range(4):
				score = fitter2000.fitProfileToItself(p.profile[side],q.profile[side],p.sidetype[side],q.sidetype[side])
				if (score > cutoffScore): break
				diffscore += score
			else:
				match = [q.id, diffscore, geoscore]
				matches += [match]

	matches = sorted(matches,key=lambda x: x[1])
	if (len(matches) == 0): print ('-- No Matches --')
	for i,elem in enumerate(matches):
		print( i,': ', elem[0], '  \t(',elem[1],')', '  \t(',elem[2],')' )

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