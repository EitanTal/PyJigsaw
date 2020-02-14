import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
import cv2
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
	#return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
	return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

def dewhite(img):
	sy,sx = img.shape
	work = img[:]
	
	for y in range(sy):
		for x in range(sx):
			p = work[y][x]
			if (p > 128):    work[y][x] = 255
	
	mask = make_floodfill_mask(work, 0, 0)

	#[mask==0] and [img==0]
	for y in range(sy):
		for x in range(sx):
			if (mask[y][x] != 0):
				img[y][x] = 0
				
	for y in range(sy):
		for x in range(sx):
			img[y][x] = 255-img[y][x]
	
	return img

def deexposure(img):
	# de expose: 10x10 corner average should be 222.
	avg = 0
	for y in range(10):
		for x in range(10):
			avg += img[y][x]
	avg = avg / 100
	desired = 222
	factor = desired / avg
	img = img * factor
		
	return img
	
def clearbg(img):
	# top    25% is Background
	# bottom 25% is a peice inside
	# everything else is midrange
	sy, sx = img.shape
	
	#de-shadow:
	# midrange starts after histogram intensity goes below 0.05%
	# pixels that fall inside this range will be clipped to 64...192
	# pixels that fall outside this range will be set to 0 or 255.
	hist = np.histogram(img, bins=range(256))[0]
	actualTop = 0
	peakStarted = False
	peakpct = 0.01 * sy * sx     #  1%
	thrspct = 0.0005 * sy * sx   #  0.05%
	for i in range(250,-1,-1):
		if (hist[i] > peakpct): peakStarted = True
		if (peakStarted and hist[i] < thrspct): 
			actualTop = i
			break

	actualBottom = 0
	peakStarted = False
	for i in range(0,250):
		if (hist[i] > peakpct): peakStarted = True
		if (peakStarted and hist[i] < thrspct): 
			actualBottom = i
			break
			
	if (debugPreProcessing): print ('Actual top/bottom ranges: ', (actualBottom*100/256), (actualTop*100/256) )

	for y in range(sy):
		for x in range(sx):
			p = img[y][x]
			if (p < actualBottom or p < bottom): img[y][x] = 0
			if (p > actualTop    or p > top):    img[y][x] = 255

	return img
	
def midrange2grey(work):
	#work = np.where((work > bottom and work < top), white/2, work)
	sy, sx = work.shape
	for y in range(sy):
		for x in range(sx):
			p = work[y][x]
			work[y][x] = white/2
			if (p < bottom): work[y][x] = 0
			if (p > top):    work[y][x] = 255
	return work

def glareElimination(img):
	# plot 7x7 pixels white in the top-left corner.
	for y in range(7):
		for x in range(7):
			img[y][x] = 255
	
	# add a white column at the end:
	white  = 256
	sy, sx = img.shape
	img = np.column_stack( [ img , np.full(sy, 255, dtype=np.uint8) ] )
	
	# Convert all midrange to grey
	work = np.copy(img)
	midrange2grey(work)
	if debugPreProcessing: imshow(img)

	# Turn all black clusters to grey except the biggest one.
	cm = findCenter(work)
	cm = refinecm(work, cm)
	mask = make_floodfill_mask(work, int(cm[0]), int(cm[1]))

	#[mask==0] and [img==0]
	for y in range(sy):
		for x in range(sx):
			if (mask[y][x] == 0 and work[y][x] == 0):
				work[y][x] = white/2
				img[y][x] = white/2
	
	if debugPreProcessing: imshow(img)
	
	# Turn all grey islands within the black to black.
	mask = np.copy(work)
	floodfill(mask, 1, 1, 128)
	floodfill(mask, 1, 1, 0)
	for y in range(sy):
		for x in range(sx):
			if (mask[y][x] > bottom and mask[y][x] < top):
				work[y][x] = 0
				img[y][x] = 0
				
	if debugPreProcessing: imshow(img)
	
	# Turn all grey islands that don't touch black to white
	mask = np.copy(work)
	floodfill(mask, int(cm[0]), int(cm[1]), 128)
	floodfill(mask, int(cm[0]), int(cm[1]), 0)
	for y in range(sy):
		for x in range(sx):
			if (mask[y][x] > bottom and mask[y][x] < top):
				work[y][x] = 255
				img[y][x] = 255

	if debugPreProcessing: imshow(img)
	
	# Turn all white islands in grey to grey			
	mask = np.copy(work)
	floodfill(mask, 0, 0, 128)
	for y in range(sy):
		for x in range(sx):
			if (mask[y][x] == 255):
				work[y][x] = white/2
				img[y][x] = white/2
	
	return img
	
##############################################################

def findCenter(a):
	# find the center of mass
	sy, sx = a.shape
	x, y = np.meshgrid(np.linspace(0,sx-1,sx), np.linspace(0,sy-1,sy))
	zx = x * a
	zy = y * a
	tot =  a.sum()
	xsum = zx.sum()
	ysum = zy.sum()
	return ( xsum / tot, ysum / tot )

def refinecm(img, cm):
	print('Center of mass:', cm)
	
	_y = int(cm[1])
	_x = int(cm[0])

	# cross pattern search to find a black pixel:
	for i in range(55):
		if (img[_y+i][_x  ] == 0): return (_x  ,_y+i)
		if (img[_y-i][_x  ] == 0): return (_x  ,_y-i)
		if (img[_y  ][_x+i] == 0): return (_x+i,_y  )
		if (img[_y  ][_x-i] == 0): return (_x-i,_y  )
	
	
def inbound(img, crd):
	sy, sx = img.shape
	#print (crd,(sx,sy))
	return (crd[0] >= 0 and crd[1] >= 0 and crd[0] < sx and crd[1] < sy)
	
def makeRad(a,dmax,cm):
	degzoom = 1
	degrees = 360
	newimgSz = (dmax,degrees*degzoom) # 360 degrees, N units max.
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
			else:
				v = 0
			if (v != 0):
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
	for i in range(50): # maximum 50 iterations
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
	
	#open a 7x7 window around every point
	for i in range(4):
		x = corners[i][0]
		y = corners[i][1]
		dBest = 0
		_xBest = 0
		_yBest = 0
		for _x in range(-windowSzMinus+x,windowSzPlus+x):
			for _y in range(-windowSzMinus+y,windowSzPlus+y):
				if _y >= sy or _x >= sx: continue
				if (img[_y][_x] != 255): continue
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
		
	
def findCornersAndClass(img):
	# Find Center of mass:
	cm = findCenter(img)
	if (debugGeometry):
		print('center of mass:', cm)

	# Find Corners and Knobs:
	b = makeRad(img, 1500, cm)
	o = findMinima(-b)
	knobs = len(o) - 4

	# Find Dips:
	dipThresh = b.mean() * 0.5
	minimas = findMinima(b)
	if (debugGeometry):
		print ('Maxima:', o)
		print ('Minima:', minimas)
	
	dips = 0
	#print ('minimas: ',minimas)
	nextD = 0
	for d in minimas:
		#print ('Inspected minima: ',b[d])
		if (b[d] < dipThresh and d > nextD): 
			dips += 1
			nextD = d + 60 # Next Dip expected to be opposite of this one. This is done to avoid noise.
	
	# Deduce peice type:
	flats = 4 - knobs - dips
	
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

	# Debug prints:
	print ('knobs, dips, flats: ', (knobs, dips, flats))
	print ('Corners: ', corners)
	if (debugGeometry):
		graphshow(b) # Show the rad graph.
		img1 = img[:]
		for c in corners:
			crossSz = 5
			for t in range(-crossSz,crossSz+1):
				img1[c[1]+t,c[0]] = 255-img1[c[1]+t,c[0]]
				img1[c[1],c[0]+t] = 255-img1[c[1],c[0]+t]
		imshow(255-img1)

	return (corners, flats)

	
def bilinear(img, x, y):
	sy, sx = img.shape
	if (sy  <= y+1) or (sx <= x+1): return 0
	if (y   <  0  ) or (x  <    0): return 0
	iX = int(x)
	_X = x - iX
	nX = 1 - _X
	iY = int(y)
	_Y = y - iY
	nY = 1 - _Y
	L  = [0,0,0,0]
	L[0] = img[iY  ][iX  ]
	L[1] = img[iY  ][iX+1]
	L[2] = img[iY+1][iX  ]
	L[3] = img[iY+1][iX+1]
	a1 = L[0] * nX + L[1] * _X
	a2 = L[2] * nX + L[3] * _X
	a  = a1   * nY + a2   * _Y
	return a
	
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
		x, y = np.meshgrid(np.linspace(0,sx-1,sx), np.linspace(-hsy,hsy-1,sy))
		_x = x * v1[0] + y * v2[0] + a[0]
		_y = x * v1[1] + y * v2[1] + a[1]
		

		for tx in range(sx):
			for ty in range(sy):
				xx = _x[ty][tx]
				yy = _y[ty][tx]
				tmp = bilinear(img,xx,yy)
				if (tmp < 0.25*white): tmp = 0
				if (tmp > 0.75*white): tmp = 255
				b[ty][tx] = tmp
				

		########## post processing ##########
		percent5  = int(sx*0.05)
		percent95 = int(sx*0.95)
		for tx in chain(range(percent5), range(percent95, sx)):
			hit = 5
			for ty in range(sy):
				if (hit > 0):
					if (b[ty][tx] >  0): hit = hit -1				
					if (b[ty][tx] == 255): hit = 0
				if (hit == 0):
					b[ty][tx] = 255
					
		mask = np.copy(b)
		midrange2grey(mask)
		floodfill(mask, 0, 0, 128)
		floodfill(mask, 0, 0, 127)
		b[mask!=127]=255
		
		########## depth and shape analysis #################
		mindepth = sy
		maxdepth = 0
		for tx in range(sx):
			for ty in range(sy):			
				if (b[ty][tx] != 255): maxdepth = max(maxdepth, ty)
				if (b[ty][tx] != 0): mindepth = min(mindepth, ty)
		pct_max = 100*maxdepth/sy
		pct_min = 100*mindepth/sy
		if ((pct_max - pct_min) < 5):     theType =  0
		elif ((pct_max + pct_min) > 100): theType = -(sy-maxdepth)
		else:                             theType =  (sy-mindepth)

		if debug:
			imshow(255-b)
		
		p.append(b)
		types.append(theType)
	return (p, types)

def imgPreprocessing(InputImg):
	img = rgb2gray(InputImg)
	if (True): #breakme
		vimg = dewhite(img)
	else:
		img = deexposure(img)

	img = clearbg(img)
	imshow(img)
	#img = glareElimination(img)
	
	
	if debug:
		imshow(img)
	# invert:
	img = 255 - img	
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
		corners, flats = findCornersAndClass(img)
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

fname = '1_1'


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
