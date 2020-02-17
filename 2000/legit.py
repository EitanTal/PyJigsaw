import numpy as np
import sys
import jigsaw
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

workdir = r'C:\jigsaw\data\2000'


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

def convert_fname(fname):
	if '.'  in fname: fname = fname[:-4]
	return fname

def check(id):
	fname = convert_fname(id)
	j = jigsaw.jigsaw.load(fname)
	angtolerance_warn = 10
	angtolerance_fail = 13
	jang = np.array(j.ang)
	f1 = jang < 90-angtolerance_fail
	f2 = jang > 90+angtolerance_fail
	w1 = jang < 90-angtolerance_warn
	w2 = jang > 90+angtolerance_warn
	fail = np.logical_or(f1,f2).any()
	warn = np.logical_or(w1,w2).any()
	if fail:
		print ('FAIL', fname, j.ang, ' !!! strange angles !!!')
	elif warn:
		print ('WARN', fname, j.ang, ' !!! strange angles !!!')
	else:
		pass#print (' OK ', fname, j.ang )

def checkfolder():
	for filename in os.listdir(workdir+'\\npz'):
		if filename.endswith(".npz"): 
			 check(filename)

if __name__ == '__main__':
	fname = 'a1a'
	if len(sys.argv) >= 2: 
		fname = sys.argv[1]
		check(fname)
	elif (len(sys.argv) == 1):
		checkfolder()
