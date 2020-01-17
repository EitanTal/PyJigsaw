import numpy as np
import sys
import jigsaw
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


workdir = r'C:\jigsaw\data'


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

if __name__ == '__main__':
	fname = 'a1a'
	if len(sys.argv) >= 2: fname = sys.argv[1]
	if '.npz' in fname or '.jpg' in fname: fname = fname[:-4]
	j = jigsaw.jigsaw.load(fname)
	angtolerance = 7
	for ang in j.ang:
		if ang < 90-angtolerance or ang > 90+angtolerance:
			print ('FAIL', fname, j.ang, ' !!! strange angles !!!')
			exit(1)
	print (' OK ', fname, j.ang )

	
