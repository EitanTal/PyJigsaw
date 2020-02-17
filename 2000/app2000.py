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

######### THREAD STUFF #############
threadList = ["Thread-1", "Thread-2", "Thread-3","Thread-4", "Thread-5", "Thread-6"]
exitFlag = 0
queueLock = threading.Lock()
workQueue = queue.Queue(0)
resultQueue = queue.Queue(0)

cutoffScore = 3000
cutoffTotal = 10000

class myThread (threading.Thread):
   def __init__(self, threadID, name, q):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = name
      self.q = q
   def run(self):
      #print ("Starting " + self.name)
      process_data(self.name, self.q)
      #print ("Exiting " + self.name)

def process_data(threadName, q):
	while not exitFlag:
		queueLock.acquire()
		if not workQueue.empty():
			data = q.get()
			queueLock.release()
			p_profile, q_profile, p_gauge, q_gauge, qid, rot, geoscore = data
			print ("%s processing %s" % (threadName, qid))
			fitscore = fitter2000.fitProfileToItself(p_profile, q_profile, p_gauge, q_gauge)
			if (fitscore > cutoffScore): fitscore = cutoffTotal
			result = [qid, rot, fitscore, geoscore]
			queueLock.acquire()
			resultQueue.put(result)
			queueLock.release()
		else:
			queueLock.release()
		time.sleep(0.01)
################################


def main():
	database = getdb()
	while True:
		wait4histogram()
		delay()
		img = capture()
		p = processor2000.process_cam(img)
		x = findit(p, database)
		if (x): saveAndRemove(x, img, database)

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
	filename = r"C:\jigsaw\2000\rgb.png"
	#filename = r"C:\jigsaw\data\2000\t1.jpg"
	return cv2.imread(filename)

def typeAgrees(t1, t2):
	if (t1 == 0) and (t2 == 0): return True
	if (t1 > 0) and (t2 > 0): return True
	if (t1 < 0) and (t2 < 0): return True
	return False

def findit(p, database):
	cutoffLenScore = 64
	cutoffAngScore = 10
	cutoffGauge = 15
	cutoffGeoScore = 200

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
	scoreSoFar = {}
	#cutoffScore = 3000
	#cutoffTotal = 10000
	for side in range(4):
		# Create threads.
		threads = []
		global exitFlag
		exitFlag = 0
		for threadID, tName in enumerate(threadList):
			thread = myThread(threadID+1, tName, workQueue)
			thread.start()
			threads.append(thread)

		# (LOCK jobs queue)
		queueLock.acquire()
		jobs = []
		for qid,geoscore,rot in geomatch:
			full_id = qid + '@' + str(rot)
			if (full_id in scoreSoFar.keys() and scoreSoFar[full_id] > cutoffTotal): continue
			for q in database:
				if (qid != q.id): continue
				diffscore = 0
				p.orient(rot)
				job = [np.copy(p.profile[side]),q.profile[side],p.sidetype[side],q.sidetype[side], q.id, rot, geoscore]
				jobs += [job]
				workQueue.put(job)
		queueLock.release()
		# (UNLOCK)

		# (RUN populate job_results)
		if (0):
			job_results = []
			for j in jobs:
				p_profile, q_profile, p_gauge, q_gauge, qid, rot, geoscore = j
				fitscore = fitter2000.fitProfileToItself(p_profile, q_profile, p_gauge, q_gauge)
				if (fitscore > cutoffScore): fitscore = cutoffTotal
				result = [qid, rot, fitscore, geoscore]
				job_results += [result]

		while not workQueue.empty():
   			time.sleep(1) #pass
		exitFlag = 1
		for t in threads:
			t.join()		
		# (END-RUN. WAIT FOR FINISH)

		# (PARSE RESULTS)
		matches = []
		#for jr in job_results:
		while not resultQueue.empty():
			jr = resultQueue.get()
			qid, rot, fitscore, geoscore = jr
			full_id = qid + '@' + str(rot)
			ssf = 0 if full_id not in scoreSoFar.keys() else scoreSoFar[full_id]
			ssf += fitscore
			scoreSoFar[full_id] = ssf

	for qid,geoscore,rot in geomatch:
		full_id = qid + '@' + str(rot)
		diffscore = scoreSoFar[full_id] 
		if (diffscore < cutoffTotal):
			match = [qid, diffscore, geoscore]
			matches += [match]

	matches = sorted(matches,key=lambda x: x[1])
	if (len(matches) == 0): print ('-- No Matches --')
	for i,elem in enumerate(matches):
		print( i,': ', elem[0], '  \t(',elem[1],')', '  \t(',elem[2],')' )

	cmdline = input(">").strip()
	if (cmdline.isnumeric()):
		return matches[int(cmdline)][0]
	elif cmdline == 'q':
		exit(1)
	else:
		return None

def saveAndRemove(x, img, database):
	cv2.imwrite(datadir + '\\' + 'cam' + '\\' + x + '.png', img)
	for j in database:
		if (j.id == x):
			database.remove(j)
			break

main()