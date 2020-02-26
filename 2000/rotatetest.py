# Rotate-test
from cv2 import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def imshow(a):
	plt.imshow(a, cmap = plt.get_cmap('gray'))
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

in_filename = 'testgrid.png'
gridtest = cv2.imread(in_filename)
i1 = rotate(gridtest, -0.2)
i2 = rotate(gridtest, -0.1)
i3 = rotate(gridtest,  0.1)
i4 = rotate(gridtest,  0.2)

cv2.imwrite('i1'+in_filename, i1)
cv2.imwrite('i2'+in_filename, i2)
cv2.imwrite('i3'+in_filename, i3)
cv2.imwrite('i4'+in_filename, i4)
