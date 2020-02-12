import numpy as np
import matplotlib.image as mpimg
import sys

workdir = r'C:\jigsaw\data\2000'
edgepct = 0.60

def rotate(l, n):
    return l[n:] + l[:n]
	
def rotate_p(l, n):
	result = l
	for i in range(n):
		result = rotate_one(result)
	return result

def rotate_one(l):
	a = l[0]
	b = l[1]
	c = l[2]
	d = l[3]
	return [b,c,d,a]

def LoadBigImg():
	#filename = workdir+'\\'+'1200ffs.png'
	filename = workdir+r'\boxart\edited'+r'\clearart.png' # 600 dpi
	imgTmp = mpimg.imread(filename)
	print('Loaded:', filename)
	return imgTmp
	
def DoSingle(i, x, y):
	print(i.shape)
	sy, sx = i.shape
	print(i.shape)
	px = sx / 50
	py = sy / 40
	startx = px*x
	starty = py*y
	
	topx = int(startx - (px*edgepct))
	topy = int(starty - (py*edgepct))
	endx = int(startx + (px*(1+edgepct)))
	endy = int(starty + (py*(1+edgepct)))
	
	# clip to image range
	topx = max(topx, 0)
	topy = max(topy, 0)
	endx = min(endx, sx-1)
	endy = min(endy, sy-1)
	
	print('working on',x,y,'crds:',topx,topy,endx,endy)
	z = i[topy:endy,topx:endx]
	filename = r'\%d_%d.png'%(x,y)
	wd = workdir+r'\breakme'
	filename = wd+filename
	mpimg.imsave(filename, z)
	print('saved:',filename)
	
def rgba2grey(rgba):
	return np.dot(rgba[...,:3], [0.299, 0.587, 0.114])
	
if __name__ == '__main__':
	rgba = LoadBigImg()
	i = rgba2grey(rgba)
	
	if ('-all' in sys.argv):
		for y in range(40):
			for x in range(50):
				DoSingle(i,x,y)
	else:
		x = y = 0
		if (len(sys.argv) >= 3):
			x = int(sys.argv[1])
			y = int(sys.argv[2])
		DoSingle(i,x,y)
		