import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import colors
import numpy as np
import math
import jigsaw2000
import sys
from cv2 import cv2

def imshow(a):
	plt.imshow(a, cmap = plt.get_cmap('gray'))
	plt.show()


def visualise(r):
	# make a color map of fixed colors
	A = 'red'            # Black-on-black
	B = 'black'          # Grey-on-black 
	D = 'brown'          # Grey-on-grey (nearly non-scoring)
	W = 'white'          # Gap
	C = 'lightgrey'      # Grey-on-white  (good / non-scoring)		
	G = 'lightgrey'         # Black-on-white (good / non-scoring)

	X = 'black'          # Unused

	cmap = colors.ListedColormap(
		[ X,A,B,X,G,X,B,X,D,X,C,X,G,C,X,X,W])
	bounds=[0, 8.5, 9.5, 10.5, 11.5, 12.5,   16.5, 17.5, 18.5, 19.5, 20.5, 32.5, 33.5, 34.5, 35.5, 36.5]
	norm = colors.BoundaryNorm(bounds, cmap.N)

	# tell imshow about color map so that only set colors are used
	img = plt.imshow(r, interpolation='nearest', 
						cmap=cmap, norm=norm)

	# make a color bar
	plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds, ticks=[9,   10,         12,            17,   18,         20,      33,     34,         36])
	plt.show()

def rotate(grayimg, degs):
    white = 256

    theta = np.deg2rad(degs)
    ang1 = 0
    ang2 = np.deg2rad(90)

    hsx = grayimg.shape[1]

    pt1 = (hsx, 0)
    pt2 = (hsx+math.cos(ang1), math.sin(ang1))
    pt3 = (hsx+math.cos(ang2), math.sin(ang2))

    ang1 += theta
    ang2 += theta

    dpt1 = (hsx, 0)
    dpt2 = (hsx+math.cos(ang1), math.sin(ang1))
    dpt3 = (hsx+math.cos(ang2), math.sin(ang2))

    dpt = np.float32((dpt1, dpt2, dpt3))
    pt = np.float32((pt1, pt2, pt3))

    M = cv2.getAffineTransform(pt,dpt)

    grayimg2 = cv2.resize(grayimg, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
    b = 255-cv2.warpAffine(255-grayimg2,M,(grayimg2.shape[0],grayimg2.shape[1]))
    c = cv2.resize(b, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    result = np.zeros(grayimg.shape)
    result[:] = 128
    result[c < 0.25*white] = 0
    result[c > 0.75*white] = 255
    return result

class Fitter():
	class score:
		def __ge__(self, other):
			in1 = self.val()
			in2 = other.val()
			return (in1 >= in2)

		def __lt__(self, other):
			in1 = self.val()
			in2 = other.val()
			return (in1 < in2)

		def __add__(self, other):
			if (other.sx == -1) or (self.sx == -1):
				self.sx = -1
			else:
				self.blackOnBlack += other.blackOnBlack
				self.greyOnBlack += other.greyOnBlack
				self.greyOnGrey += other.greyOnGrey
				self.whiteOnWhite += other.whiteOnWhite
				self.nudge = [0,0]
				self.degs = 0
				self.sx += other.sx
				self.defaultNudge = [0,0]
			return self

		def __init__(self, maxNudge):
			self.sx = 0
			self.final = False
			self.blackOnBlack = 0
			self.greyOnBlack = 0
			self.greyOnGrey = 0
			self.whiteOnWhite = 0
			self.nudge = [0,0]
			self.degs = 0
			self.defaultNudge = [0,0]
			self.maxNudge = maxNudge

		@staticmethod
		def infinite():
			a = Fitter.score(0)
			a.sx = -1
			return a

		def __str__(self):
			if (self.sx == 0): return '----'
			if (0):
				mystr = 'BB{}\tGB{}\tWW{}\tGG{}\tD:{:.1f}\tN:[{} {}]\tV:{}\tD:[{} {}]\t'.format(
					self.blackOnBlack, self.greyOnBlack, self.whiteOnWhite, self.greyOnGrey, self.degs, self.nudge[0], self.nudge[1], int(self.val()), self.defaultNudge[0], self.defaultNudge[1])
			elif (0):
				mystr = 'BB{}\tGB{}\tWW{}\tGG{}\tD:{:.1f}\tN:[{} {}]\t'.format(
					self.blackOnBlack, self.greyOnBlack, self.whiteOnWhite, self.greyOnGrey, self.degs, self.nudge[0], self.nudge[1])
			else:
				mystr = 'BB{}\tGB{}\tWW{}\tGG{}\t'.format(
					self.blackOnBlack, self.greyOnBlack, self.whiteOnWhite, self.greyOnGrey )

			return mystr

		def val(self):
			if (self.sx < 0): return 999999
			if (self.sx == 0): return 0
			if (self.final and (abs(self.nudge[0]) >= self.maxNudge or abs(self.nudge[1]) >= self.maxNudge)): return 999998
			diffscore = 0
			diffscore += self.blackOnBlack*10
			diffscore += self.greyOnBlack*0.3
			diffscore += self.greyOnGrey*0.1
			diffscore += self.whiteOnWhite*0.066
			return 100 * diffscore / self.sx

		def finalize(self):
			self.final = True

	def __init__(self):
		self.Debug      = False
		self.DebugShow  = False
		self.BruteForce = False
		self.StepsDebug = False
		self.maxNudge = 15
		self.prefetch = {}
		self.bestScore = 200

	def _clearPrefetch(self):
		self.prefetch = {}

	def QuickYNudge(self, a, gA, gB):
		syA, sxA = a.shape
		hA = gA
		hB = -gB
		yNudge = syA - (hA + hB)
		if (self.StepsDebug):
			print ('Height of knob:', hA, '@ y=', syA - hA)
			print ('Height of Dip:', hB, '@ y=', hB)
			print ('Height of A:', syA)
			print ('yNudge:', yNudge)

		# gauge is measured that the knob doesn't include the gray pixels.
		# only the dip includes the gray pixels.
		# we don't want gray-on-black, and preferably not gray-on-gray either.
		# apply -6 to compensate for gray-on-black and maybe some gray-on-gray.
		# the fidgeting algorithm will find the exact best spot.
		return int(yNudge - 6)

	def fit(self, a, b, ga, gb, ga_x=0, gb_x=0):
		if (self.BruteForce):
			if (ga > 0):
				return self.fitProfilesBruteForce(a, b, ga, gb)
			else:
				return self.fitProfilesBruteForce(b, a, gb, ga)
		else:
			if (ga > 0):
				TheScore = self.fitProfiles_step1(a, b, ga, gb, ga_x, gb_x) # a is the knob, b is the dip.
			else:
				TheScore = self.fitProfiles_step1(b, a, gb, ga, gb_x, ga_x) # b is the knob, a is the dip.

		TheScore.finalize()
		self.bestScore = min(TheScore.val(), self.bestScore)
		return TheScore

	def fitProfiles_step1(self, a, b, ga, gb, ga_x, gb_x):
		# run without a rotation:
		baseScore = self.fitProfiles_step2(a, b, ga, gb, ga_x, gb_x)
		if (baseScore.sx == -1): return baseScore
		if (baseScore.val() > self.bestScore * 2): 
			if (self.Debug): print ('fit cut short.')
			return baseScore  # applying degree search reduces score by up to 66%. If the score is double, there's no chance.

		goodNudge = baseScore.nudge
		origscore = baseScore.val()

		# Test 2 directions:
		b180_orig = np.rot90(b, 2) # rotate profile b by 180 degrees
		testdirs = [0.2, -0.2]
		for direction in testdirs:
			newDeg = direction
			b180 = rotate(b180_orig, newDeg)
			self._clearPrefetch()
			newVal = self.fitProfiles_step3(a, b180, goodNudge)
			if (newVal < baseScore):
				break
		else:
			if (self.Debug): print ( 100,'%' ,'Before: ', origscore)
			return baseScore

		# Keep going in this direction until the result is worse.  (5 ticks total, up to +/- 1.0 degs)
		if (self.StepsDebug): print ('Run!')
		bestDeg = newDeg
		bestVal = newVal
		for _ in range(4):
			newDeg += direction
			b180 = rotate(b180_orig, newDeg)
			self._clearPrefetch()
			newVal = self.fitProfiles_step3(a, b180, goodNudge)
			if (newVal < bestVal):
				bestVal = newVal
				bestDeg = newDeg
			else:
				break

		if (self.Debug):
			finalscore = bestVal.val()
			pct = int(finalscore * 100 / origscore)
			print ( pct,'%' ,'Before: ', origscore, 'After:', finalscore, 'deg:', bestDeg)
		bestVal.degs = bestDeg
		bestVal.defaultNudge = baseScore.defaultNudge
		return bestVal

	def fitProfiles_step2(self, a, b, ga, gb, ga_x, gb_x):
		self._clearPrefetch()

		# B side adjustment:
		b180 = np.rot90(b, 2) # rotate profile b by 180 degrees
		yNudge = self.QuickYNudge(a, ga, gb)
		xNudge = int(ga_x - gb_x)
		
		defaultNudge = [xNudge, yNudge] # X: + is right, - is left. Y: + is down, - is up
		self.defaultNudge = defaultNudge
		if (self.Debug):
			print ('DefaultNudge:', defaultNudge)
		nudge = np.array(defaultNudge)
		result = self.fitProfiles_step3(a, b180, nudge)
		result.defaultNudge = defaultNudge
		return result
		
	def fitProfiles_step3(self, a, b180, nudge):
		baseScore = self.fitProfiles_internal(a, b180, nudge)
		while True:
			result = self.descent(baseScore, nudge, a, b180, True)
			if (result):
				nudge = result[0]
				baseScore = result[1]
			else:
				break

		if (self.Debug):
			print( 'Nudge:', nudge)
			print( 'Fit rating:', baseScore.val())
			self.fitProfiles_internal(a, b180, nudge, self.DebugShow)

		return baseScore

	def descent(self, curScore, curNudge, a, b180, downAllowed=False):
		if (self.StepsDebug): print ('Descent! @', curNudge, curScore.val())
		left  = np.array([-1, 0])
		right = np.array([1, 0])
		up    = np.array([0, -1])
		down  = np.array([0,  1])

		# Test 4 directions:
		testdirs = [left, right, up]
		if (downAllowed): testdirs += [down]
		for direction in testdirs:
			newNudge = curNudge + direction
			newVal = self.fitProfiles_internal(a, b180, newNudge)
			if (newVal < curScore):
				break
		else:
			return False

		# Keep going in this direction until the result is worse.
		if (self.StepsDebug): print ('Run!')
		bestNudge = np.copy(newNudge)
		bestVal = newVal
		for _ in range(30):
			newNudge += direction
			newVal = self.fitProfiles_internal(a, b180, newNudge)
			if (newVal < bestVal):
				bestNudge = np.copy(newNudge)
				bestVal = newVal
			else:
				return bestNudge, bestVal

	def fitProfiles_internal(self, a, b180, nudge, show=False):
		# a and b might be of a slightly different size.
		if not show and str(nudge) in self.prefetch: return self.prefetch[str(nudge)]
		if (self.StepsDebug): print ('*', nudge)

		if (abs(nudge[0]) > self.maxNudge or abs(nudge[1]) > self.maxNudge):
			return Fitter.score.infinite()

		b = np.copy(b180)
		syA, sxA = a.shape
		syB, sxB = b.shape
		sx = min(sxA, sxB)
		#sy = min(syA, syB)

		# resulting image:
		r = np.zeros([syA, sxA], dtype=np.uint8)
		r[a==0]               = 1
		r[(a!=255) & (a!=0)]  = 2
		r[a==255]             = 4

		# Bit Meaning:
		# Bit 0 (1) - A pixel (Black)
		# Bit 1 (2) - A pixel (Midrange)
		# Bit 2 (4) - A pixel (White)
		
		# Bit 3 (8) - B pixel (Black)
		# Bit 4 (16)- B pixel (Midrange)
		# Bit 5 (32)- B pixel (White)

		# perform nudge on b: example nudge = (5, -6)   # X: + is right, - is left. Y: + is down, - is up
		# x:
		nx = int(nudge[0])
		ny = int(nudge[1])
		if (nx > 0):
			# don't actually nudge b. erase columns from r instead.
			r = r[:,nx:]
		elif (nx < 0):
			# erase colums from b:
			b = b[:,-nx:]
			
		# y:
		if (ny > 0):
			# don't actually nudge b. erase rows from r instead.
			r = r[ny:,:]
		elif (ny < 0):
			# erase rows from b:
			b = b[-ny:,:]

		# clip b & r to agree on size:
		syR, sxR = r.shape
		b = b[:syR,:sxR]
		syB, sxB = b.shape
		r = r[:syB,:sxB]

		r[b==0]              |= 8
		r[(b!=255) & (b!=0)] |= 16
		r[b==255]            |= 32
		
		# scoring:
		# Bits 0 and 3 will score +1 overlap.                                ( 9)
		# Bits 1 and 3 will score +1 overlap. (less severe black-on-gray)    (10)
		# Bits 0 and 4 will score +1 overlap. (less severe black-on-gray)    (17)
		# Bits 5 and 2 will score +1 gap.                                    (36)
		# White-on-midrange is allowed
		# Midrange-on-midrange is allowed.
		# Bits 012 without 345 is allowed. (nudges, offsets, size differences, etc)
		result = Fitter.score(self.maxNudge)

		# cut off some of the extreme columns to the sides as the rotation can distort the result
		r = r[:,3:-3]

		result.blackOnBlack = np.sum((r &  9) ==  9)
		result.greyOnBlack  = np.sum((r & 10) == 10) + np.sum((r & 17) == 17)
		result.whiteOnWhite = np.sum((r & 36) == 36)
		result.greyOnGrey   = np.sum((r & 18) == 18)
		result.nudge        = np.copy(nudge)
		result.sx = sx-6

		# Score normalize:
		if (self.StepsDebug):
			print( 'Fit rating:', result.val())
		#if (StepsDebug or show):
		if (show):
			visualise(r)

		self.prefetch[str(nudge)] = result
		return result

	def fitProfilesBruteForce(self, a,b,ga,gb):
		print ('!')
		b180 = np.rot90(b, 2) # rotate profile b by 180 degrees
		TopScore = score.infinite()
		bestNudge = []
		searchradius = 15

		self._clearPrefetch()
		for y in range(-searchradius,searchradius+1):
			for x in range(-searchradius,searchradius+1):
				nudge = [x,y]
				_score = self.fitProfiles_internal(a, b180, np.array(nudge))
				if (_score < TopScore):
					TopScore = _score
					bestNudge = nudge[:]
		
		print ('TopScore', TopScore.val()*100, 'Nudge:', bestNudge)
		return TopScore

	def interactive(self, profile1, profile1_orientation, profile2, profile2_orientation):
		nudge = [0,0]
		j1 = jigsaw2000.jigsaw.load(profile1)
		j1.orient(profile1_orientation)
		j2 = jigsaw2000.jigsaw.load(profile2)
		j2.orient(profile2_orientation)	
		# B side adjustment:
		a = j1.profile[0]
		b = j2.profile[0]
		b180_orig = np.rot90(b, 2) # rotate profile b by 180 degrees
		deg = 0
		b180 = b180_orig

		while True:
			cmd = input('command?')
			if cmd == '8':
				nudge[1] = nudge[1] -1
			elif cmd == '2':
				nudge[1] = nudge[1] +1
			elif cmd == '4':
				nudge[0] = nudge[0] -1
			elif cmd == '6':
				nudge[0] = nudge[0] +1
			elif cmd == 'e':
				deg += 0.2
				print ('deg =',deg)
				b180 = rotate(b180_orig, deg)
				self._clearPrefetch()
			elif cmd == 'q':
				deg -= 0.2
				print ('deg =',deg)
				b180 = rotate(b180_orig, deg)
				self._clearPrefetch()
			elif cmd == 'r':
				_score = self.fitProfiles_internal(a, b180, nudge)
				result = self.descent(_score, nudge, a, b180, True)
				if (result):
					nudge = result[0]
			
			_score = self.fitProfiles_internal(a, b180, nudge, cmd == '?')

			print (_score.val(), _score)

if __name__ == '__main__':
	profile1 = 'Other2/h_5'      # !!!! KNOB !!!!
	profile1_orientation = 3
	profile2 = 'Other3/a_4'      # !!!! DIP  !!!!
	#profile2 = 'Other3/h_1'      # !!!! DIP  !!!! WRONG ONE
	profile2_orientation = 1
	
	f = Fitter()

	#f.interactive(profile1, profile1_orientation, profile2, profile2_orientation)

	if len(sys.argv) >= 5:
		profile1 = sys.argv[1]
		profile1_orientation = int(sys.argv[2])
		profile2 = sys.argv[3]
		profile2_orientation = int(sys.argv[4])
	
	j1 = jigsaw2000.jigsaw.load(profile1)
	j1.orient(profile1_orientation)
	
	j2 = jigsaw2000.jigsaw.load(profile2)
	j2.orient(profile2_orientation)

	#f.BruteForce = True
	score = f.fit(j1.profile[0], j2.profile[0], j1.sidetype[0], j2.sidetype[0], j1.gauge_x[0], j2.gauge_x[0])
	print(score, score.val())
	
