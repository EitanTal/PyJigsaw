import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
from cv2 import cv2
import jigsaw
import sys
from itertools import chain

class Quad:
	pass

class xjigsaw:
	pass

debug = False
debugPreProcessing = False
debugGeometry = False
conernsoverride = False
allowedOreintation = None
white = 256
bottom = 0.25 * white
top    = 0.75 * white
cornerdb = {}
datadir = r'C:\jigsaw\data\2000\breakme'

################# service functions #########################
	
def smooth(x,window_len=11,window='hanning'):
	if window_len<3:
		return x
	s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
	#print(len(s))
	if window == 'flat': #moving average
		w=np.ones(window_len,'d')
	else:
		w=eval('np.'+window+'(window_len)')
	y=np.convolve(w/w.sum(),s,mode='valid')
	return y

def imshow(a):
	plt.imshow(a, cmap = plt.get_cmap('gray'))
	plt.show()

def graphshow(a):
	plt.plot(a)
	plt.show()

def rangeconvert(img, lower, upper, set):
	work = np.copy(img)
	lr = work >= lower
	gr = work <= upper
	m  = np.logical_and(gr,lr)
	work[m] = set
	return work

def make_floodfill_mask(img, x, y):
	matrix_np = np.asarray(img).astype(np.uint8)
	mask = np.zeros(np.asarray(img.shape)+2, dtype=np.uint8)
	start_pt = (x,y)
	cv2.floodFill(matrix_np, mask, start_pt, 255, flags=4)
	mask = mask[1:-1, 1:-1]
	return mask

def floodfill(img, x, y, c):
	mask = make_floodfill_mask(img, x, y)
	img[mask != 0] = c
	
#################### image pre-processing #####################
	
def rgb2gray(rgb):
	return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

def midrange2grey(work):
	gr = work > top
	lr = work < bottom

	work[:] = 128
	work[gr] = 255
	work[lr] = 0

	return work
	
##############################################################

def findCenter(a):
	# find the center of mass
	sy, sx = a.shape
	blacksPixels = cv2.inRange(a, 0, 1)
	x, y = np.meshgrid(np.linspace(0,sx-1,sx), np.linspace(0,sy-1,sy))
	zx = x * blacksPixels
	zy = y * blacksPixels
	tot =  blacksPixels.sum()
	xsum = zx.sum()
	ysum = zy.sum()
	return ( xsum / tot, ysum / tot )
	
def inbound(img, crd):
	sy, sx = img.shape
	#print (crd,(sx,sy))
	return (crd[0] >= 0 and crd[1] >= 0 and crd[0] < sx and crd[1] < sy)
	
def makeRad(a,dmax,cm):
	degzoom = 1
	degrees = 360
	#newimgSz = (dmax,degrees*degzoom) # 360 degrees, N units max.

	b = np.zeros(degrees*degzoom)
	for ang in range(degrees * degzoom):
		ang_ = math.radians(ang/degzoom)
		x = math.cos(ang_)
		y = math.sin(ang_)
		for d in reversed(range(dmax)):
			x1 = int(x * d + cm[0])
			y1 = int(y * d + cm[1])
			if (inbound(a,(x1,y1))):
				v = a[y1][x1]
				if (v == 0):
					b[ang] = d
					break

	return b
	
def pad(a, num):
	mid = a[:]
	end = a[-num:]
	start = a[:num]
	res = np.concatenate([end,mid,start])
	return res
	
def findMinima(f):
	g = pad(f,15)
	a = smooth(g)
	p = int((len(a) - len(f))/2)  # actual pad
	res = []
	isMinima = np.r_[True, a[1:] <= a[:-1]] & np.r_[a[:-1] < a[1:], True]
	for (i,x) in enumerate(isMinima):
		if (x and i in range(p,360+p)): 
			res += [i-p]

	# for every minima, look to the left and right to determine width.
	# calculate the height difference.
	# if width * height is too low, this minima is noise.
	for m in res:
		minima = m+p
		minimaLeft = 0
		minimaRight = 0
		minimaV = a[minima]
		mV = minimaV
		minimaHeightLeft = 0
		minimaHeightRight = 0
		for i in range(minima,1,-1):
			v = a[i-1]
			if (v > minimaV):
				minimaV = v
				minimaLeft += 1
			else:
				minimaHeightLeft = minimaV - mV
				break
		minimaV = a[minima]
		for i in range(minima+1,len(a)):
			v = a[i]
			if (v > minimaV):
				minimaV = v
				minimaRight += 1
			else:
				minimaHeightRight = minimaV - mV
				break
		minimaIntegral = (minimaLeft*minimaHeightLeft) + (minimaRight*minimaHeightRight)

		if ( minimaIntegral < 33 ):
			print ('Discarded noise')
			res.remove(m)
			
	return res


def refineCorners(img, corners):
	for _ in range(50): # maximum 50 iterations
		previous = corners[:]
		refineCornersX(img, corners)
		if (previous == corners): break

def refineCornersX(img, corners):
	cm = [0,0] #x, y
	for i in range(4):
		cm[0] += corners[i][0]
		cm[1] += corners[i][1]
	cm[0] /= 4
	cm[1] /= 4
	
	windowSz = 11
	windowSzMinus = windowSz // 2
	windowSzPlus = windowSzMinus+1
	sy, sx = img.shape
	
	#open a square window around every point
	for i in range(4):
		x = corners[i][0]
		y = corners[i][1]
		dBest = 0
		_xBest = 0
		_yBest = 0
		for _x in range(-windowSzMinus+x,windowSzPlus+x):
			for _y in range(-windowSzMinus+y,windowSzPlus+y):
				if _y >= sy or _x >= sx: continue
				if (img[_y][_x] != 0): continue
				dx = _x - cm[0] 
				dy = _y - cm[1]
				d = dx*dx + dy*dy
				if (d > dBest):
					dBest = d
					_xBest = _x
					_yBest = _y
		if (_xBest > 0 and _yBest > 0):
			corners[i] = (_xBest, _yBest)
	
def orderByAngle(corners):
	cm = [0, 0]
	cornersWithAngle = []
	result = []
	for c in corners:
		cm[0] += c[0] / 4
		cm[1] += c[1] / 4
	for c in corners:
		vectorx = c[0] - cm[0]
		vectory = c[1] - cm[1]
		ang = ((math.atan2(vectory, vectorx)/math.pi*180)+360) % 360
		#print('v:', vectorx,vectory, 'a:', ang)
		cornersWithAngle.append([ang, c])
	record = sorted(cornersWithAngle,key=lambda x: x[0])
	for c in record:
		result.append(c[1])
	
	return result

def rejectNonCornerMaximas(records, maxdist):
	# sometimes there's maximas in the bottom of the dips. Take em out.
	records_ = []
	for r in records:
		if r[1] > 100:
			records_ += [r]
		else:
			print ('!! Rejected a suspected non-corner:', r)
	
	# Continue...
	records_ = sorted(records_,key=lambda x: x[0]) # Sort by angle
	result = []
	
	for i, r in enumerate(records_):
		prev = records_[i-1]
		next = records_[(i+1)%len(records_)]
		dist_from_prev = min(abs(prev[0] - r[0]), 360-abs(prev[0] - r[0]))
		dist_from_next = min(abs(next[0] - r[0]), 360-abs(next[0] - r[0]))
		if (dist_from_prev < maxdist and dist_from_next < maxdist):
			print ('!! Rejected a suspected non-corner:', r)
			continue
		result += [r]
	return result
		
	
def findCorners(img):
	# Find Center of mass:
	cm = findCenter(img)
	if (debugGeometry):
		print('center of mass:', cm)

	# Find Corners and Knobs:
	b = makeRad(img, 1500, cm)
	o = findMinima(-b)

	# Find Dips:
	dipThresh = b.mean() * 0.5
	minimas = findMinima(b)
	if (debugGeometry):
		print ('Maxima:', o)
		print ('Minima:', minimas)
	
	#print ('minimas: ',minimas)
	nextD = 0
	for d in minimas:
		#print ('Inspected minima: ',b[d])
		if (b[d] < dipThresh and d > nextD): 
			nextD = d + 60 # Next Dip expected to be opposite of this one. This is done to avoid noise.
	
	# Locate 4 corners cartesian coordinates:
	record = []
	for peak in o:
		record.append((peak,b[peak]))
	
	# Expect non-corners to be at most 65 degrees from BOTH their 2 neighboring maximas.
	backup = record
	record = rejectNonCornerMaximas(record, 65)
	if (len(record) < 4):
		print('FindCorners: rejectNonCornerMaximas failed.')
		record = backup
	
	# Sort by distance.
	# keep the lowest four, they are the corners.

	record = sorted(record,key=lambda x: x[1])
	# reject maximas that are too close: (less than 30 degs in each direction)
	
	del record[4:]
	
	# Great. Now sort by angle again.
	record = sorted(record,key=lambda x: x[0])
	
	corners = []
	if (len(record) < 4):
		print('FindCorners: too few records.', record)
	for i in range(4): # Four corners...
		angle, dist = record[i]
		_ang = math.radians(angle)
		x = int((math.cos(_ang) * dist) + cm[0])
		y = int((math.sin(_ang) * dist) + cm[1])
		corners.append((x,y))

	if (debugGeometry):
		print ('Candidates for corners: (Polar)', record)
		print ('Candidates for corners: (Cartesian)', corners)

	# Increase accuracy of corners' locations.
	refineCorners(img, corners)
	
	# sometimes the order of corners get messed up.
	corners = orderByAngle(corners)

	if (debugGeometry):
		print ('Candidates for corners: (Polar)', record)
		print ('Candidates for corners: (Cartesian)', corners)

	if (debugGeometry):
		graphshow(b) # Show the rad graph.
		img1 = np.copy(img)
		for c in corners:
			crossSz = 5
			for t in range(-crossSz,crossSz+1):
				img1[c[1]+t,c[0]] = 255-img1[c[1]+t,c[0]]
				img1[c[1],c[0]+t] = 255-img1[c[1],c[0]+t]
		imshow(img1)

	return corners

def makeQuad(corners):
	sideLen  = []
	sideVect = []
	sideAng  = []
	crnrAng  = []
	for i in range(4):
		c1 = corners[i]
		c2 = corners[(i+1) % 4]
		dx = c2[0] - c1[0]
		dy = c2[1] - c1[1]
		ang = math.atan2(dy,dx)/math.pi*180
		d = math.sqrt(dx*dx + dy*dy)
		sideLen.append(d)
		sideVect.append([dx, dy])
		sideAng.append(ang)
		
	for i in range(4):
		a1 = sideAng[i-1]
		a2 = sideAng[i  ]
		a = (a2 - a1 + 360) % 360
		a = 180 - a
		crnrAng.append(a)
		
	q = Quad()
	q.corners  = corners
	q.sideLen  = sideLen
	q.sideVect = sideVect
	q.sideAng  = sideAng
	q.crnrAng  = crnrAng
	return q
	
def getProfiles(img, q):
	p = []
	types = []
	for i in range(4):
		sx = int( q.sideLen[i] )
		sy = sx
		angle = q.sideAng[i]
		_ang = math.radians(angle)
		newimgSz = (sy,sx)
		b = np.zeros(newimgSz) 
		a = q.corners[i]
		v1x = math.cos(_ang)
		v1y = math.sin(_ang)
		v1 = [v1x , v1y ]
		
		v2x = math.cos(_ang + math.pi/2)
		v2y = math.sin(_ang + math.pi/2)
		v2 = [v2x , v2y ]
		hsy = sy//2
		
		pt1 = a 
		pt2 = a + np.array(v1)*sx
		pt3 = a + np.array(v2)*hsy
		pt = np.float32((pt1, pt2, pt3))

		dpt1 = (0, hsy)
		dpt2 = (sx,hsy)
		dpt3 = (0, sy )
		dpt = np.float32((dpt1, dpt2, dpt3))

		M = cv2.getAffineTransform(pt,dpt)

		b = 255-cv2.warpAffine(255-img,M,(sx,sy))
		b[b < 0.25*white] = 0
		b[b > 0.75*white] = 255

		########## post processing ##########
		percent5  = int(sx*0.05)
		percent95 = int(sx*0.95)
		for tx in chain(range(percent5), range(percent95, sx)):
			hit = 5
			for ty in range(sy):
				if (hit > 0):
					#if (b[ty][tx] <  255): hit = hit -1				
					if (b[ty][tx] == 0): hit = 0
				if (hit == 0):
					b[ty][tx] = 0
					
		mask = np.copy(b)
		mask[0,:] = 255 # Set top line of mask to white pixels to make the floodfill continuous.
		midrange2grey(mask)
		floodfill(mask, 0, 0, 128)
		floodfill(mask, 0, 0, 255)
		floodfill(mask, 0, 0, 127)
		b[mask!=127]=0
		
		########## depth and shape analysis #################
		mindepth = sy
		maxdepth = 0
		for tx in range(sx):
			for ty in range(sy):			
				if (b[ty][tx] != 0): maxdepth = max(maxdepth, ty)
				if (b[ty][tx] == 0): mindepth = min(mindepth, ty)
		pct_max = 100*maxdepth/sy
		pct_min = 100*mindepth/sy
		if ((pct_max - pct_min) < 5):     theType =  0
		elif ((pct_max + pct_min) > 100): theType = -(sy-maxdepth)
		else:                             theType =  (sy-mindepth)

		if debug:
			imshow(b)
		
		p.append(b)
		types.append(theType)
	return (p, types)

def imgPreprocessing(InputImg):
	img = rgb2gray(InputImg)
	return img
	
def process(imgnr):
	filename = datadir+'\\'+str(imgnr)+".png"
	imgTmp = cv2.imread(filename)
	img = imgPreprocessing(imgTmp)
	x = xjigsaw()
	if imgnr in cornerdb.keys():
		corners = cornerdb[imgnr]
		print('manual corners entered for', imgnr)
	else:
		corners = findCorners(img)
	q = makeQuad(corners)
	p = getProfiles(img, q)
	flats = sum(theType == 0 for theType in p[1])
	print( 'actual flats:', flats )
	
	x.corners = corners
	x.flats = flats
	x.q = q
	x.types = p[1]
	x.p = p[0]
	x.id = imgnr
	return x
	
def export(xjig):
	print(xjig.q.sideLen)
	print(xjig.types)
	print(xjig.q.crnrAng)
	j = jigsaw.jigsaw(xjig.q.sideLen, xjig.types, xjig.q.crnrAng, xjig.p, allowedOreintation, xjig.id)
	j.save()
	
def example(i):
	print ('Analysing', i)
	x = process(i)
	export(x)

fname = '9_9'


if '-debug' in sys.argv:
	debug = True
if '-debugall' in sys.argv:
	debug = True
	debugPreProcessing = True
	debugGeometry = True
if '-debugpre' in sys.argv:
	debugPreProcessing = True
if '-debuggeo' in sys.argv:
	debugGeometry = True

if len(sys.argv) >= 2: 
	fname = sys.argv[1]
if '.jpg' in fname: fname = fname[:-4]
example(fname.replace('\\','/'))
