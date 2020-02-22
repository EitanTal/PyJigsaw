import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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
		print( 'Fit rating:', int(lastScore*100))
		fitProfiles_internal(a, b180, nudge, True)

	return int(lastScore*100)

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
	r[a==255]             = 1
	r[(a!=255) & (a!=0)]  = 2
	r[a==0]               = 4

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

	r[b==255]            |= 8
	r[(b!=255) & (b!=0)] |= 16
	r[b==0]              |= 32
	
	# scoring:
	# Bits 0 and 3 will score +1 overlap.                                ( 9)
	# Bits 1 and 3 will score +1 overlap. (less severe black-on-gray)    (10)
	# Bits 0 and 4 will score +1 overlap. (less severe black-on-gray)    (17)
	# Bits 5 and 2 will score +1 gap.                                    (36)
	# White-on-midrange is allowed
	# Midrange-on-midrange is allowed.
	# Bits 012 without 345 is allowed. (nudges, offsets, size differences, etc)
	
	score = 0
	score += np.sum((r &  9) ==  9)
	score += np.sum((r & 10) == 10)
	score += np.sum((r & 17) == 17)
	score += np.sum((r & 36) == 36)

	# Score normalize:
	score = score / sx
	if (StepsDebug):
		print( 'Nudge:', nudge)
		print( 'Fit rating:', int(score*100))
	if (StepsDebug or show):
		imshow(r)
	return score
