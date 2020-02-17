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

def QuickYNudge(a, b, gA, gB):
	# p: the pixel that the tip of the gauge touches, be it the tip of the knob or the pit of the dip
	pa = a.shape[0] - gA
	pb = b.shape[0] - gB

	# after nudge, the two images must line up on P.
	# nudge is a vector applied to B.
	# pa = pb + N
	# N = pa - pb
	yNudge = pa-pb

	return int(yNudge)

def fitProfileToItself(cam, boxart, cam_g, boxart_g, _debug=False):
	yNudge = QuickYNudge(cam, boxart, cam_g, boxart_g)
	
	maxNudge = 15

	defaultNudge = [0, yNudge] # X: + is right, - is left. Y: + is down, - is up
	nudge = np.array(defaultNudge)
	
	left = np.array([-1, 0])
	right = np.array([1, 0])
	up    = np.array([0, -1])
	down  = np.array([0, 1])
	
	#check
	baseScore = fitProfiles_internal(cam, boxart, nudge)
	lastScore = baseScore
	
	# try moving left:
	newScore = fitProfiles_internal(cam, boxart, nudge+left)
	# good? continue.
	if (newScore < baseScore):
		lastScore = baseScore
		while (newScore < lastScore):
			nudge = nudge + left
			lastScore = newScore
			newScore = fitProfiles_internal(cam, boxart, nudge+left)
			if (abs(nudge[0]) >= maxNudge): break
	else: # Bad? try moving the other direction
		newScore = fitProfiles_internal(cam, boxart, nudge+right)
		if (newScore < baseScore):
			lastScore = baseScore
			while (newScore < lastScore):
				nudge = nudge + right
				lastScore = newScore
				newScore = fitProfiles_internal(cam, boxart, nudge+right)
				if (abs(nudge[0]) >= maxNudge): break

	# try moving down:
	newScore = fitProfiles_internal(cam, boxart, nudge+down)
	if (newScore < lastScore):
		lastScore = lastScore
		while (newScore < lastScore):
			nudge = nudge + down
			lastScore = newScore
			newScore = fitProfiles_internal(cam, boxart, nudge+down)

	if (Debug or _debug):
		print( 'Nudge:', nudge)
		print( 'Fit rating:', int(lastScore*100))
		fitProfiles_internal(cam, boxart, nudge, True)

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
	# Black pixel from A on a black pixel from B:   0
	# Black pixel from A on a grey  pixel from B:   1
	# Black pixel from A on a white pixel from B:   3
	# White pixel from A on a black pixel from B:   3
	# White pixel from A on a grey  pixel from B:   0
	# White pixel from A on a white pixel from B:   0
	# White pixel from A on a N/A   pixel from B:   0


	# Bits 0 and 4 will score +1 overlap.     (1+16)
	# Bits 0 and 5 will score +3 overlap.     (1+32)
	# Bits 2 and 3 will score +3 overlap.     (4+8)
	# Bits 012 without 345 is allowed. (nudges, offsets, size differences, etc)
	
	score = 0
	score += np.sum((r & 17) == 17)
	score += (np.sum((r & 33) == 33)*3)
	score += (np.sum((r & 12) == 12)*3)

	# Score normalize:
	score = score / sx
	if (StepsDebug):
		print( 'Nudge:', nudge)
		print( 'Fit rating:', int(score*100))
	if (StepsDebug or show):
		imshow(r)
	return score
