import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import colors
import numpy as np
import math
import jigsaw
import sys

Debug  = False
StepsDebug = False

def imshow(a):
	plt.imshow(a, cmap = plt.get_cmap('gray'))
	plt.show()

def QuickYNudge(a, gA, gB):
	syA, sxA = a.shape
	hA = gA
	hB = -gB
	yNudge = syA - (hA + hB)
	if (StepsDebug):
		print ('Height of knob:', hA, '@ y=', syA - hA)
		print ('Height of Dip:', hB, '@ y=', hB)
		print ('Height of A:', syA)
		print ('yNudge:', yNudge)
	return int(yNudge)

class score:
	def __ge__(self, other):
		in1 = self.val()
		in2 = other.val()
		return (in1 >= in2)

	def __lt__(self, other):
		in1 = self.val()
		in2 = other.val()
		return (in1 < in2)

	def __init__(self):
		self.sx = 0

	@staticmethod
	def infinite():
		a = score()
		a.sx = -1
		return a

	def __str__(self):
		if (self.sx == 0): return '----'
		mystr = 'BB{}\tGB{}\tWW{}\tGG{}\tN:[{} {}]'.format(self.blackOnBlack, self.greyOnBlack, self.whiteOnWhite, self.greyOnGrey, self.nudge[0], self.nudge[1])
		return mystr

	def val(self):
		if (self.sx < 0): return 999999
		if (self.sx == 0): return 0
		diffscore = 0
		diffscore += self.blackOnBlack*1
		#diffscore += self.greyOnBlack*0.03
		#diffscore += self.greyOnGrey*0.01
		#diffscore += self.whiteOnWhite/3
		return int( 100 * diffscore / self.sx )


def fitProfilesEx(a, b, ga, gb, _debug=False):
	# B side adjustment:
	b180 = np.rot90(b, 2) # rotate profile b by 180 degrees
	yNudge = QuickYNudge(a, ga, gb)
	
	maxNudge = 15

	defaultNudge = [0, yNudge] # X: + is right, - is left. Y: + is down, - is up
	nudge = np.array(defaultNudge)
	
	left = np.array([-1, 0])
	right = np.array([1, 0])
	up    = np.array([0, -1])
	
	#check
	baseScore = fitProfiles_internal(a, b180, nudge)
	lastScore = baseScore
	
	# try moving left:
	newScore = fitProfiles_internal(a, b180, nudge+left)
	# good? continue.
	if (newScore < baseScore):
		lastScore = baseScore
		while (newScore < lastScore):
			nudge = nudge + left
			lastScore = newScore
			newScore = fitProfiles_internal(a, b180, nudge+left)
			if np.any(abs(nudge) >= maxNudge): break
	else: # Bad? try moving the other direction
		newScore = fitProfiles_internal(a, b180, nudge+right)
		if (newScore < baseScore):
			lastScore = baseScore
			while (newScore < lastScore):
				nudge = nudge + right
				lastScore = newScore
				newScore = fitProfiles_internal(a, b180, nudge+right)
				if np.any(abs(nudge) >= maxNudge): break

	# try moving up:
	newScore = fitProfiles_internal(a, b180, nudge+up)
	if (newScore < lastScore):
		lastScore = lastScore
		while (newScore < lastScore):
			nudge = nudge + up
			lastScore = newScore
			newScore = fitProfiles_internal(a, b180, nudge+up)

	if (Debug or _debug):
		print( 'Nudge:', nudge)
		print( 'Fit rating:', lastScore.val())
		fitProfiles_internal(a, b180, nudge, True)

	return lastScore

def fitProfiles_internal(a, b180, nudge, show=False):
	# a and b might be of a slightly different size.
	if (StepsDebug): print ('*')
	b = np.copy(b180)
	syA, sxA = a.shape
	syB, sxB = b.shape
	sx = min(sxA, sxB)
	sy = min(syA, syB)

	# resulting image:
	r = np.zeros([syA, sxA], dtype=np.uint8)
	r[a==0]             = 1
	r[(a!=255) & (a!=0)]  = 2
	r[a==255]               = 4

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

	r[b==0]            |= 8
	r[(b!=255) & (b!=0)] |= 16
	r[b==255]              |= 32
	
	# scoring:
	# Bits 0 and 3 will score +1 overlap.                                ( 9)
	# Bits 1 and 3 will score +1 overlap. (less severe black-on-gray)    (10)
	# Bits 0 and 4 will score +1 overlap. (less severe black-on-gray)    (17)
	# Bits 5 and 2 will score +1 gap.                                    (36)
	# White-on-midrange is allowed
	# Midrange-on-midrange is allowed.
	# Bits 012 without 345 is allowed. (nudges, offsets, size differences, etc)
	result = score()

	result.blackOnBlack = np.sum((r &  9) ==  9)
	result.greyOnBlack  = np.sum((r & 10) == 10) + np.sum((r & 17) == 17)
	result.whiteOnWhite = np.sum((r & 36) == 36)
	result.greyOnGrey   = np.sum((r & 18) == 18)
	result.nudge = nudge
	result.sx = sx

	# Score normalize:
	if (StepsDebug):
		print( 'Nudge:', nudge)
		print( 'Fit rating:', result.val())
	if (StepsDebug or show):
		# make a color map of fixed colors
		A = 'cyan'
		B = 'yellow'
		C = 'magenta'
		D = 'brown'
		W = 'white'
		G = 'purple'
		X = 'black'

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
		#imshow(r)

	return result

# a is the knob, b is the dip.
def fitProfiles(a, b, ga, gb, _debug=False):
	if (ga > 0):
		return fitProfilesEx(a, b, ga, gb, _debug)
	else:
		return fitProfilesEx(b, a, gb, ga, _debug)

def fitProfilesBruteForce(a,b,ga,gb):
	# B side adjustment:
	b180 = np.rot90(b, 2) # rotate profile b by 180 degrees
	
	TopScore = score.infinite()
	#TopScore.sx = -1
	bestNudge = []

	searchradius = 15

	#check
	for y in range(-searchradius,searchradius+1):
		for x in range(-searchradius,searchradius+1):
			nudge = [x,y]
			_score = fitProfiles_internal(a, b180, nudge)
			if (_score < TopScore):
				TopScore = _score
				bestNudge = nudge
	
	#print ('TopScore', TopScore.val()*100, 'Nudge:', bestNudge)
	print (TopScore)


if __name__ == '__main__':
	Debug = True
	profile1 = 'Other2/e_1'
	profile1_orientation = 3
	profile2 = 'Other3/f_1'
	profile2_orientation = 1
	
	if len(sys.argv) >= 5:
		profile1 = sys.argv[1]
		profile1_orientation = int(sys.argv[2])
		profile2 = sys.argv[3]
		profile2_orientation = int(sys.argv[4])
	
	j1 = jigsaw.jigsaw.load(profile1)
	j1.orient(profile1_orientation)
	
	j2 = jigsaw.jigsaw.load(profile2)
	j2.orient(profile2_orientation)
	
	fitProfiles(j1.profile[0], j2.profile[0], j1.sidetype[0], j2.sidetype[0])
	#fitProfilesBruteForce(j1.profile[0], j2.profile[0], j1.sidetype[0], j2.sidetype[0])
	