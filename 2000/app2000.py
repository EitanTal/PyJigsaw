import os
import sys
import jigsaw2000
import fitter2000 as fitter
import math
from puzzlemap import ThePuzzleMap
from parse import parse

sx = 50
sy = 40

datadir = r'C:\jigsaw\data\2000\npz'

Debug = False
Mirror = False
ConsiderOrientation = False
ExhaustiveSrch = False

SolvingType = 'Spiral'
#SolvingType = 'Diamond'
#SolvingType = 'Raster'
#SolvingType = 'Special'

def clear():
	os.system('cls')

def calcscore(list):
	if (len(list) == 0): return 0
	if (len(list) == 1): return list[0]
	sum = 0
	for a in list:
		sum += a*a
	return math.sqrt(sum)

def calc_fit_score(fitscores):
	if (len(fitscores) == 0): return 0
	if (len(fitscores) == 1): return fitscores[0]

	s = fitter.Fitter.score(0)
	for a in fitscores:
		s += a
	return s


class Board:
	def __init__(self, sizeX, sizeY):
		self.board = []
		line = [('?',0)] * sizeX
		for _ in range(sizeY):
			self.board += [line[:]]
			
	def cursor(self,x,y):
		self.board[y][x] = ('_',0)
		
	def update(self,x,y,p):
		self.board[y][x] = (p[0],p[1])
		if (valid(p[0])): orient(p[0], p[1])
	
	def show(self):
		clear()
		for line in self.board:
			c = ''
			reverse = -1 if Mirror else 1
			for char in line[::reverse]:
				if (char[0] == '?'):
					c += ' '
				elif (char[0] == '_'):
					c += '?'
				else:
					c += '#'
			print (c)
			
	def dump(self):
		with open('board2.txt', 'w') as f:
			for line in self.board:
				for char in line:
					f.write (str(char)+'\n')
				
	def load(self):
		with open('board2.txt', 'r') as f:
			for y in range(sy):
				for x in range(sx):
					txt = f.readline()
					p = eval(txt)
					self.update(x,y,p)

	def at(self,x,y):
		if (x < 0) or (y < 0) or (x >= sx) or (y >= sy):
			return None
		return self.board[y][x]


def valid(x):
	return x not in ['?', '_']

b = Board(sx,sy) # Start with an empty board

all_inventory = None

def getUnusedInventory():
	global all_inventory
	if (all_inventory is None):
		all_inventory = populateInventory()
	result = all_inventory[:]
	for y in range(sy):
		for x in range(sx):
			p = b.at(x,y)
			if (p[0] in result):
				result.remove(p[0])
	return result

def populateInventory():
	folderlist = []

	for f in os.listdir(datadir):
		if (os.path.isdir(datadir+'\\'+f)):
			folderlist += [f]

	npz = []

	for f in folderlist:
		if f.strip() != '':
			npz += populateInventoryFolder(f)
	
	return npz

def populateInventoryFolder(folder):
	all = os.listdir(datadir+'\\'+folder)
	npz = []
	for a in all:
		if a.endswith('.npz'): npz.append(folder+'/'+a[:-4])
	return npz
	
loadedpcs = {}
	
def getjigsaw(peice):
	if peice not in loadedpcs.keys():
		x = jigsaw2000.jigsaw.load(peice)
		loadedpcs[peice] = x
	return loadedpcs[peice]

def getj(oriented):
	j = getjigsaw(oriented[0])
	j.orient(oriented[1])
	return j
	
def getp(j, side):
	return (j.profile[side], j.sidetype[side], j.gauge_x[side])
	
def fitProfiles(f, a, b):
	if (Debug): print (a[1], b[1])
	if (a[1] == 0 or b[1] == 0):
		score = 6000 # Attempted to match a flat.
	elif (a[1] * b[1] < 0):
		score = f.fit(a[0], b[0], a[1], b[1], a[2], b[2])
		if (Debug): print( 'fitter rating = ', score, 'for', a[1], b[1] )
	else:
		score = 5000 # Wrong combination.
	return score

def orient(peice, orientation):
	j = getjigsaw(peice)
	j.orient(orientation)
	return j

def getCandidates(nUp, nDn, nRt, nLt,n7,n9,n1,n3, tq, exhaustive=False):
	jUp = jDn = jRt = jLt = None # 4 neighboring jigsaw pieces, if available.

	p0 = p1 = p2 = p3 = None # 4 profiles a candidate should match with.
	l0 = l1 = l2 = l3 = None # 4 legths a candidate should match with.
	a0 = a1 = a2 = a3 = None # 4 angles a candidate should match with.
	
	flats = 0 # amount of flat profiles in the surrounding area.

	fit0 = fitter.Fitter()
	fit1 = fitter.Fitter()
	fit2 = fitter.Fitter()
	fit3 = fitter.Fitter()
	for fit in (fit0,fit1,fit2,fit3):
		if (exhaustive): fit.maxNudge = 30
		if (Debug): fit.Debug = True
		#fit.Debug = True
	
	# profiles (or flat profiles), lengths
	if ( nUp == None ):
		p2 = 'flat'
		flats += 1
	elif ( valid(nUp[0]) ):
		jUp = getj(nUp)
		p2 = getp(jUp, 0)
		l2 = jUp.sidelen[0]
		
	if ( nDn == None ):
		flats += 1
		p0 = 'flat'
	elif ( valid(nDn[0]) ):
		jDn = getj(nDn)
		p0 = getp(jDn, 2)
		l0 = jDn.sidelen[2]
		
	if ( nRt == None ):
		flats += 1
		p3 = 'flat'
	elif ( valid(nRt[0]) ):
		jRt = getj(nRt)
		p3 = getp(jRt, 1)
		l3 = jRt.sidelen[1]
		
	if ( nLt == None ):
		flats += 1
		p1 = 'flat'
	elif ( valid(nLt[0]) ):
		jLt = getj(nLt)
		p1 = getp(jLt, 3)
		l1 = jLt.sidelen[3]

	j1 = j3 = j7 = j9 = None
	if n1 and valid(n1[0]): j1 = getj(n1)
	if n3 and valid(n3[0]): j3 = getj(n3)
	if n7 and valid(n7[0]): j7 = getj(n7)
	if n9 and valid(n9[0]): j9 = getj(n9)

	# angles:
	if (jUp and jLt and j7):
		a2 = 360 - j7.ang[0] - jLt.ang[3] - jUp.ang[1]
	if (jUp and jRt and j9):
		a3 = 360 - j9.ang[1] - jRt.ang[2] - jUp.ang[0]
	if (jDn and jRt and j3):
		a0 = 360 - j3.ang[2] - jRt.ang[1] - jDn.ang[3]
	if (jDn and jLt and j1):
		a1 = 360 - j1.ang[3] - jLt.ang[0] - jDn.ang[2]
		
	if (flats == 1):
		if (jLt):
			if (nUp == None): # clockwise solving
				a2 = 180 - jLt.ang[3]
			else: # counter-clockwise solving
				a1 = 180 - jLt.ang[0]
		if (jUp):
			if (nRt == None): # clockwise solving
				a3 = 180 - jUp.ang[0]
			else:
				a2 = 180 - jUp.ang[1]
		if (jRt):
			if (nDn == None): # clockwise solving
				a0 = 180 - jRt.ang[1]
			else: # counter-clockwise solving
				a3 = 180 - jRt.ang[2]
		if (jDn):
			if (nLt == None): # clockwise solving
				a1 = 180 - jDn.ang[2]
			else: # counter-clockwise solving
				a0 = 180 - jDn.ang[3]

	if (Debug): print ('angs:', a0, a1, a2, a3)
	
	inventory = getUnusedInventory()
	# Geometry match:
	matches = []
	for j in inventory:
		p = orient(j, 0)
		if p.determinetype() != tq: continue
		
		for i in range(4):
			p = orient(j, i)
			if (ConsiderOrientation):
				if not p.isOrientationAllowed(i): continue
			angular_score = []
			legnth_score  = []

			
			if (a0 is not None): angular_score += [abs(p.ang[0] - a0)]
			if (a1 is not None): angular_score += [abs(p.ang[1] - a1)]
			if (a2 is not None): angular_score += [abs(p.ang[2] - a2)]
			if (a3 is not None): angular_score += [abs(p.ang[3] - a3)]
			
			if (l0 is not None): legnth_score += [abs(p.sidelen[0] - l0)]
			if (l1 is not None): legnth_score += [abs(p.sidelen[1] - l1)]
			if (l2 is not None): legnth_score += [abs(p.sidelen[2] - l2)]
			if (l3 is not None): legnth_score += [abs(p.sidelen[3] - l3)]
			
			#disagreement on flats?
			if (type(p0) == str) ^ (p.sidetype[0] == 0 ): continue
			if (type(p1) == str) ^ (p.sidetype[1] == 0 ): continue
			if (type(p2) == str) ^ (p.sidetype[2] == 0 ): continue
			if (type(p3) == str) ^ (p.sidetype[3] == 0 ): continue
			
			#if (Debug): print ('p angs:', p.ang[0], p.ang[1], p.ang[2], p.ang[3])
			#if (Debug): print ('p lens:', p.sidelen[0] ,p.sidelen[1] , p.sidelen[2] ,p.sidelen[3] )
			if (Debug): print (angular_score, legnth_score)
			
			legnth_score_avg = calcscore(legnth_score)
			angular_score_avg = calcscore(angular_score)
			geoScore = (angular_score_avg*5 + legnth_score_avg)*10
			if (Debug): print ('score for id', p.id, 'orentation', i, 'ang score:', angular_score_avg, 'legnth_score', legnth_score_avg)
			
			if exhaustive:
				cutoffLenScore = 250
				cutoffAngScore = 50
				cutoffGeoScore = 2500
			else:
				cutoffLenScore = 16
				cutoffAngScore = 5
				cutoffGeoScore = 250
			
			if ( legnth_score_avg > cutoffLenScore ): continue
			if ( angular_score_avg > cutoffAngScore ): continue
			if ( geoScore > cutoffGeoScore ): continue

			matches.append([(p.id, i), geoScore, legnth_score_avg, angular_score_avg])
	# sort...
	matches = sorted(matches,key=lambda x: x[1])
	
	# profile match
	maxmatches = 1984
	for i, m in enumerate(matches):
		if (i > maxmatches):
			m[1] += 3800
			continue
			
		p = orient(m[0][0], m[0][1])

		if (p.sidetype[0] == 0): p0 = None
		if (p.sidetype[1] == 0): p1 = None
		if (p.sidetype[2] == 0): p2 = None
		if (p.sidetype[3] == 0): p3 = None
		if (type(p0) is str): p0 = None
		if (type(p1) is str): p1 = None
		if (type(p2) is str): p2 = None
		if (type(p3) is str): p3 = None
		
		pScore = []
		if (Debug): print('fitting for', p.id)
		if (p0 is not None): pScore += [fitProfiles(fit0, p0, getp(p, 0))]
		if (p1 is not None): pScore += [fitProfiles(fit1, p1, getp(p, 1))]
		if (p2 is not None): pScore += [fitProfiles(fit2, p2, getp(p, 2))]
		if (p3 is not None): pScore += [fitProfiles(fit3, p3, getp(p, 3))]
		result = calc_fit_score(pScore)
		if type(result) is int:
			result = fitter.Fitter.score(0)
		m[1] = result.val()
		m += [result]
	
	# Sort again...
	cutoffScore = 4000
	matches = sorted(matches,key=lambda x: x[1])
	result = []
	for m in matches:
		if (m[1] <= cutoffScore):
			result.append(m)

	return result

def niceprint(list, verbose = False):
	if (len(list) == 0): print ('-- No Matches --')
	for i,elem in enumerate(list):
		if verbose:
			print( i,': ', elem[0], '  \t(',elem[1],')', '  \t(',elem[2],elem[3],')' )
		else:
			print( i,': ', elem[0], '  \t(',elem[1],')' )
	
def choose(candidates):
	niceprint(candidates[:4]) # print top 4
	while (1):
		cmdline = input(">").strip()
		if (cmdline == ''):
			continue
		if (cmdline == 's') or (cmdline == 'dump'):
			b.dump()
			continue
		if (cmdline == 'm'): # show more candidates
			niceprint(candidates, True)
			continue
		if (cmdline == 'b'): # backward
			return None
		if (cmdline == 'q'):
			exit(1)
		if (cmdline.isnumeric()):
			return candidates[int(cmdline)][0]
		else:
			return cmdline

def getFwdDir(pos):
	x = pos[0]
	y = pos[1]
	hsx = sx // 2
	hsy = sy // 2
	#quadrants:     12
	#               34
	
	onEdge = (x == 0) or (y == 0) or (x+1 == sx) or (y+1 == sy)
	
	if (SolvingType == 'Special'):
		SpecialSolve = '' # // ! Not implemented
		SpecialSolveX = SpecialSolve.splitlines()[1:]
		c = SpecialSolveX[y][x]
		d = None
		if c == '>': d = ( 1, 0)
		if c == '<': d = (-1, 0)
		if c == 'v': d = ( 0, 1)
		if c == '^': d = ( 0,-1)
	if (SolvingType == 'Spiral' or onEdge):
		if   (x  < hsx) and (y  < hsy): # Quadrant 1
			if (x+1 >= y) : d = ( 1, 0)
			else:         d = ( 0,-1)
		elif (x >= hsx) and (y  < hsy): # Quadrant 2
			_x = (sx-1) - x # _x = distance from last x
			if (_x > y) : d = ( 1, 0)
			else:         d = ( 0, 1)
		elif (x  < hsx) and (y >= hsy): # Quadrant 3
			_y = (sy-1) - y # _y = distance from last y
			if (_y >= x) : d = ( 0,-1)
			else:          d = (-1, 0)
		else: # Quadrant 4
			_x = (sx-1) - x # _x = distance from last x
			_y = (sy-1) - y # _y = distance from last y
			if (_x >= _y) : d = (-1, 0)
			else:           d = ( 0, 1)
	elif (SolvingType == 'Raster'):
		if   (x  % 2) == 1: # Odd lines
			if (x == sx): d = ( 0, 1) # Down at the END.
			else:  d = (1, 0)         # Right
		else: # Even lines
			if (x == 1): d = ( 0, 1) # Down at the BEGINNING.
			else:  d = (-1, 0)       # Left
	elif (SolvingType == 'Diamond'):
		pass
	return d
		
def moveFwd(pos):
	# start at (0,0)
	if (pos == None):
		pos = (0,0)
		return pos

	x = pos[0]
	y = pos[1]
	#hsx = sx // 2
	hsy = sy // 2

	# end? can't move forward?
	if (SolvingType == 'Spiral'):
		if ( y == x + 1 ) and ( y == hsy ):
			return None
	elif (SolvingType == 'Raster'):
		if ( y == sy - 2 ) and ( x == sx - 2 ):
			return None

	# Find which way to move forward.
	d = getFwdDir(pos)
	if (SolvingType == 'Special' and d is None): return None
	
	# Apply the move
	pos = (x + d[0],y + d[1])
	return pos

def moveBwd(pos):
	prev_pos = None
	while (pos != prev_pos):
		prev_pos_1 = prev_pos
		prev_pos = moveFwd(prev_pos)
	return prev_pos_1

def solve():
	pos = moveFwd(None)
	while (pos):
		print ('!', pos)
		if (valid(b.at(pos[0],pos[1])[0])):
			pos = moveFwd(pos)
			continue
		b.cursor(pos[0],pos[1])
		b.show()
		# find 8-adjacent neighbors: Out of bounds are None, unsolved are ('?', 0)
		nUp = b.at(pos[0],pos[1]-1)
		nDn = b.at(pos[0],pos[1]+1)
		nRt = b.at(pos[0]+1,pos[1])
		nLt = b.at(pos[0]-1,pos[1])
		n7 = b.at(pos[0]-1,pos[1]-1)
		n9 = b.at(pos[0]+1,pos[1]-1)
		n1 = b.at(pos[0]-1,pos[1]+1)
		n3 = b.at(pos[0]+1,pos[1]+1)
		
		# Determine matching peice by the quadrant: 
		targetq = ThePuzzleMap.splitlines()[pos[1] + 1][pos[0]]
		
		candidates = getCandidates(nUp, nDn, nRt, nLt,n7,n9,n1,n3, targetq, ExhaustiveSrch)
		r = choose(candidates)
		if (r.__class__ is str):
			if r.startswith('j'):
				x = y = -1
				formats = ['j{:d},{:d}','j{:d} {:d}','j {:d},{:d}','j {:d} {:d}']
				for f in formats:
					z = parse(f, r)
					if z is not None:
						x,y = z
						break
				if (x >= 0):
					print ('Moved cursor to',  x,y)
					b.update(pos[0],pos[1],('?',0))
					pos = x,y
				else:
					print('format not recognised')
				input("Hit enter to continue").strip()
				continue
			if r == '?':
				print( 'Cursor at: ', pos )
				input("Hit enter to continue").strip()
				continue
			elif r.startswith('?'):
				x,y = parse('?{:d},{:d}', r)
				print ('peice at', x,y, 'is', b.at(x,y))
				input("Hit enter to continue").strip()
				continue
			else:
				r = eval(r)
		if (r):
			b.update(pos[0],pos[1],r)
			pos = moveFwd(pos)
		else:
			b.update(pos[0],pos[1],('?',0))
			pos = moveBwd(pos)
			print ('reverted', b.at(pos[0],pos[1]))
			input("Hit enter to continue").strip()
			b.update(pos[0],pos[1],('?',0))


def main():
	if '-new' not in sys.argv:
		b.load()
		
	if '-ext' in sys.argv:
		ExhaustiveSrch = True

	try:
		solve()
	finally:
		b.dump()
		

if __name__ == '__main__':
	main()