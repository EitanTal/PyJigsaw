import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


workdir = r'C:\jigsaw\data'


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
		self.profile = rotate_p(self.profile, orientation)
		self.ang = rotate(self.ang, orientation)
		self.north = (self.north + orientation) % 4
		
	def isOrientationAllowed(self, orientation):
		if self.allowedOreintation is None: return True
		else: return (self.allowedOreintation[orientation] > 0)
		
	def save(self):
		outfile = workdir + '\\' + str(self.id)
		orientation = self.allowedOreintation
		if orientation is None: orientation = [1,1,1,1]
		meta = np.array(self.sidelen + self.ang + self.sidetype + orientation)
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
		allowedOreintation = (meta[12:16].tolist() if (len(meta) >= 16 ) else None)
		profile = loaded['p']
		return jigsaw(sidelen, sidetype, ang, profile, allowedOreintation, id)

	def show(self, showimage):
		print('Lengths:', self.sidelen)
		print('Types:  ', self.sidetype)
		print('Angles: ', self.ang)
		print('id:     ', self.id)
		print('Orient: ', self.north)
		print('Allowed:', self.allowedOreintation)
		if (showimage):
			for p in self.profile:
				plt.imshow(p, cmap = plt.get_cmap('gray'))
				plt.show()

	@staticmethod
	def loadAll():
		pass # return a list
		
	def __init__(self, sidelen, sidetype, ang, profile, allowedOreintation, id):
		self.sidelen = sidelen
		self.sidetype = sidetype
		self.profile = profile
		self.ang     = ang
		self.id      = id
		self.north   = 0
		self.allowedOreintation = allowedOreintation

if __name__ == '__main__':
	fname = 'a1a'
	if len(sys.argv) >= 2: fname = sys.argv[1]
	if '.npz' in fname: fname = fname[:-4]
	j = jigsaw.load(fname)
	j.show('-text' not in sys.argv)
