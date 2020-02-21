import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
from cv2 import cv2
import jigsaw
import sys
from itertools import chain
import os

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

scalefactor = 0.9005 # 90.05%
scalefactorCam = 1.000
#scalefactorCam = 1.035  # make camera bigger by 3.5%

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
	
def refinecm(img, cm):
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

def findNearestElem(elem, vals):
	best = vals[0]
	bestdist = abs(elem-vals[0])
	for v in vals:
		dist = abs(elem - v)
		if (dist < bestdist):
			bestdist = dist
			best = v
	return best

def findNearestDistance(elem, vals):
	k = findNearestElem(elem, vals)
	return abs(k-elem)

def findDistanceOnCircle(elem, vals):
	circ = []
	for v in vals:
		circ.append(v-360)
		circ.append(v)
		circ.append(v+360)
	
	d = findNearestDistance(elem, circ)
	return d

def selectDiagonalConers(records):
	angles = []
	for r in records:
		if r[1] > 100:
			angles += [r[0]]
	a = [0,0,0,0]
	a[0] = findNearestElem(45, angles)
	a[1] = findNearestElem(135, angles)
	a[2] = findNearestElem(225, angles)
	a[3] = findNearestElem(315, angles)
	
	# duplicates?
	if len(set(a)) != 4: #FAIL
		print('!! selectDiagonalConers FAIL')
		return records
		
	result = []
	for r in records:
		if r[0] in a:
			result += [r]
	return result
	

def rejectNonCornerMaximas(records, maxdist):
	# sometimes there's maximas in the bottom of the dips. Take em out.
	records_ = []
	angles = []
	for r in records:
		if r[1] > 100:
			records_ += [r]
			angles += [r[0]]
		else:
			print ('!! Rejected a suspected non-corner:', r,'for being inside a dip')
	
	# Continue...
	records_ = sorted(records_,key=lambda x: x[0]) # Sort by angle

	# Special case for a + peice: it will have 8 results. four corners and four knobs. Select the ones closest to diagonals.
	if len(records_) == 8:
		print ('!! Encountered a suspicuos + shape. Guessing corners that line up with diagonals')
		first = records_[0][0]
		if (first < 25): return records_[1::2]
		else: return records_[0::2]

	# If this is a corner, then 3 other corners should be at +90, +180 and +270 roughly speaking, as peices are mostly square-ish.
	result = []
	for r in records_:
		ang = r[0]
		c2 = findDistanceOnCircle(ang+90, angles)
		c3 = findDistanceOnCircle(ang+180, angles)
		c4 = findDistanceOnCircle(ang-90, angles) # 270
		# Do not tolerate if more than 20 degrees away
		limit = maxdist
		if ((c2 < limit) and (c3 < limit) and (c4 < limit)):
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
	recordBackup = record[:]
	tryThese = [25, 20, 35, 18, 37]
	#tryThese = [20]

	for t in tryThese:
		record = recordBackup[:]
		tried = rejectNonCornerMaximas(record, t)
		if (len(tried) == 4):
			record = tried
			if (debugGeometry):	print ('rejectNonCornerMaximas success at', t)
			break
		else:
			if (debugGeometry):	print ('rejectNonCornerMaximas fail at', t,'records:',len(tried))
	else:
		print('!! rejectNonCornerMaximas failed.')
		record = selectDiagonalConers(recordBackup)
		
		#record = recordBackup[:]
	
	
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
		print ('(pre refine) Candidates for corners: (Polar)', record)
		print ('(pre refine) Candidates for corners: (Cartesian)', corners)

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
	
def getProfiles(img, q, sf):
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

		dpt1 = (0, hsy*sf)
		dpt2 = (sx*sf,hsy*sf)
		dpt3 = (0, sy*sf )
		dpt = np.float32((dpt1, dpt2, dpt3))

		M = cv2.getAffineTransform(pt,dpt)

		b = 255-cv2.warpAffine(255-img,M,(int(sx*sf),int(sy*sf)))
		b[b < 0.25*white] = 0
		b[b > 0.75*white] = 255

		# update sx, sy:
		sx = b.shape[0]
		sy = sx

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

def clearbg(rgb):
	hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
	h,s,v = cv2.split(hsv)
	b,g,r = cv2.split(rgb)
	
	zeros = np.zeros(v.shape, np.uint8)
	image = zeros[:]
	

	mask_grey  = cv2.inRange(g, 0, 80)
	mask_black = cv2.inRange(s, 0, 128)
	mask_white = cv2.inRange(s, 160, 255)
	
	# Rendering order:
	# Gray background
	# White
	# Black
	# Grey
	image[:] = 128
	cv2.bitwise_or(image, 255, dst=image, mask=(mask_white))
	cv2.bitwise_and(image, 0, dst=image, mask=(mask_black))
	cv2.bitwise_and(image, 0, dst=image, mask=(mask_grey))
	cv2.bitwise_or(image, 128, dst=image, mask=(mask_grey))

	return image

	
def glareElimination(img):
	# find center of mass
	cm = findCenter(img)
	cm = refinecm(img, cm)

	# Step 1: Clean up any white islands inside grey zone
	# Step 2: Clean up any grey islands inside white zone

	# --Step 1--
	# Fill the perim with gray
	work  = np.copy(img)
	work[0,:] = 255
	work[:,0] = 255
	work[-1,:] = 255
	work[:,-1] = 255

	# Fill the center and the perimeter with gray.
	mask1 = make_floodfill_mask(work, 0, 0)
	mask2 = make_floodfill_mask(work, int(cm[0]), int(cm[1]))
	mask3 = cv2.bitwise_or(mask1, mask2)
	img[mask3==0] = 128

	# --Step 2--
	# clear any islands
	# Turn all grey islands that don't touch black to white
	mask = np.copy(img)
	floodfill(mask, int(cm[0]), int(cm[1]), 128)
	floodfill(mask, int(cm[0]), int(cm[1]), 255)
	img[mask != 255] = 255

	return img

def Preprocessing_cam(InputImg):
	img = clearbg(InputImg)
	img = glareElimination(img)	
	return img
	
def determinetype(gauge):
	classes = {
		'00++':'-', '00-+':'-','00+-':'-','00--':'-', # corners
		'0---':'0', '0+++':'3','0--+':']','0+--':'[','0-++':']','0++-':'[','0-+-':'v','0+-+':'w', # edges
		'+-+-':'A', # classic puzzle peice
		'++--':'B', # B-type puzzle peice
		'++++':'+', # plus-type
		'----':'X', # X-type
		'+++-':'T', # T-type
		'+---':'K', # K-type
	}
	text = ''
	for g in gauge:
		if (g == 0): text += '0'
		if (g <  0): text += '-'
		if (g >  0): text += '+'
	
	t = text*2
	for k in classes.keys():
		if k in t:
			return classes[k]
	return '?'

def process_boxart(imgnr):
	filename = datadir+'\\'+str(imgnr)+".png"
	imgRgb = cv2.imread(filename)
	img = rgb2gray(imgRgb)
	x = xjigsaw()
	if imgnr in cornerdb.keys():
		corners = cornerdb[imgnr]
		print('manual corners entered for', imgnr)
	else:
		corners = findCorners(img)
	q = makeQuad(corners)
	p = getProfiles(img, q, scalefactor)
	flats = sum(theType == 0 for theType in p[1])
	#print( 'actual flats:', flats )
	
	x.corners = corners
	x.flats = flats
	x.q = q
	x.types = p[1]
	x.p = p[0]
	x.id = imgnr

	#apply scale factor to side lens:
	x.q.sideLen[0] *= scalefactor
	x.q.sideLen[1] *= scalefactor
	x.q.sideLen[2] *= scalefactor
	x.q.sideLen[3] *= scalefactor

	print('Peice Type:', determinetype(x.types), 'ID:',imgnr)
	cmd = ' '.join(['copy', r'C:\jigsaw\data\2000\breakme'+'\\'+imgnr+'.png',r'C:\jigsaw\data\2000\breakme'+'\\'+determinetype(x.types)])
	print(cmd)
	os.system(cmd)
	
	return x

def process_cam(imgTmp):
	sf = scalefactorCam
	img = Preprocessing_cam(imgTmp) # 1 Preprocessing
	corners = findCorners(img)      # 2 Corner detection
	q = makeQuad(corners)           #    (convert to quad)
	p = getProfiles(img, q, sf)     # 3 Get Profiles
	for i in range(4):	q.sideLen[i] *= scalefactorCam #apply scale factor to side lengths
	print('Peice Type:', determinetype(p[1]))
	return jigsaw.jigsaw(q.sideLen, p[1], q.crnrAng, p[0], None, 'cam')	

	
def export(xjig):
	print(xjig.q.sideLen)
	print(xjig.types)
	print(xjig.q.crnrAng)
	j = jigsaw.jigsaw(xjig.q.sideLen, xjig.types, xjig.q.crnrAng, xjig.p, allowedOreintation, xjig.id)
	j.save()
	
def example(i):
	print ('Analysing', i)
	#x = process_boxart(i)
	filename = r'C:\jigsaw\data\2000'+'\\'+i+".png"
	rgb = cv2.imread(filename)
	bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
	x = process_cam(bgr)
	x.show(True)
	x.save()

if __name__ == '__main__':
	fname = 'png/lightgreen/j_1'

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
	if '.png' in fname: fname = fname[:-4]
	example(fname.replace('\\','/'))
