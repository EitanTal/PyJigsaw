import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from cv2 import cv2
import os
import fnmatch

workdir = r'C:\jigsaw\data\2000\npz'


def rotate(l, n):
    return l[n:] + l[:n]
	
def rotate_p(l, n):
	result = l
	for _ in range(n):
		result = rotate_one(result)
	return result

def rotate_one(l):
	a = l[0]
	b = l[1]
	c = l[2]
	d = l[3]
	return [b,c,d,a]
	
class jigsaw:
	def orient(self, orientation):
		orientation = (4 + -self.north + orientation) % 4
		self.sidelen = rotate(self.sidelen, orientation)
		self.sidetype = rotate(self.sidetype, orientation)
		self.gauge_x = rotate(self.gauge_x, orientation)
		self.profile = rotate_p(self.profile, orientation)
		self.ang = rotate(self.ang, orientation)
		self.north = (self.north + orientation) % 4
			
	def save(self):
		outfile = workdir + '\\' + str(self.id)
		gauge_x = self.gauge_x
		meta = np.array(self.sidelen + self.ang + self.sidetype + gauge_x)
		profiles = self.profile
		np.savez_compressed(outfile, m=meta, p=profiles)
		
	@staticmethod
	def load(id):
		infile = workdir + '\\' + str(id) + '.npz'
		loaded = np.load(infile, allow_pickle=True)
		meta = loaded['m']
		sidelen = meta[:4].tolist()
		ang = meta[4:8].tolist()
		sidetype = meta[8:12].tolist()
		gauge_x = (meta[12:16].tolist() if (len(meta) >= 16 ) else None)
		profile = loaded['p']
		return jigsaw(sidelen, sidetype, ang, profile, gauge_x, id)

	def show(self, showimage):
		print('Lengths:', self.sidelen)
		print('Types:  ', self.sidetype)
		print('Angles: ', self.ang)
		print('id:     ', self.id)
		tmp = np.copy(self.gauge_x)
		for i in range(len(self.sidetype)):
			if (self.sidetype[i] < 0): tmp[i] = int(self.sidelen[i]) - tmp[i]
		print('gauge_x ', self.gauge_x, '(Readable:', tmp,')')
		if (showimage):
			for p in self.profile:
				plt.imshow(p, cmap = plt.get_cmap('gray'))
				plt.show()

	def exportpng(self):
		for i in range(4):
			img = self.profile[i]
			fname = self.id.replace('/','-') + '_' + str(i) + '.png'
			print( 'saved:', fname)
			cv2.imwrite(fname, img)

	@staticmethod
	def loadAll():
		pass # return a list
		
	def __init__(self, sidelen, sidetype, ang, profile, gauge_x, id):
		self.sidelen = sidelen
		self.sidetype = sidetype
		self.profile = profile
		self.ang     = ang
		self.id      = id
		self.north   = 0
		self.gauge_x = gauge_x

	def determinetype(self):
		gauge = self.sidetype
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

def unrollWildcards(fname):
	result = []
	wd = os.path.dirname(fname)
	full_wd = workdir + '\\' + wd
	fn = os.path.basename(fname)
	for w in os.listdir(full_wd):
		w = w.split('.')[0]
		if fnmatch.fnmatch(w,fn):
			result += [wd + '/' + w]

	return result

if __name__ == '__main__':
	fname = 'Other3/f_1'
	if len(sys.argv) >= 2: fname = sys.argv[1]
	if (fname == '-box'):
		workdir = r'C:\jigsaw\data\2000\npz_boxart'
		for y in range(40):
			for x in range(50):
				fname = str(x+1) + '_' + str(y+1)
				j = jigsaw.load(fname)
				print(j.determinetype(),end='')
			print()
	else:
		names = unrollWildcards(fname)
		show = (len(names) == 1)
		for x in names:
			if '.npz' in x: x = x[:-4]
			j = jigsaw.load(x)
			j.show('-text' not in sys.argv)
			if ('-png' in sys.argv): j.exportpng()
