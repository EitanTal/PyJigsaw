import cv2, time
import numpy as np
import matplotlib.image as mpimg


#video = cv2.VideoCapture(0)
a=0
Preview = True


ajpg = cv2.imread(r'C:\jigsaw\data\2000\a.jpg')
frame = cv2.cvtColor(ajpg, cv2.COLOR_RGB2BGR)
t1,t2,t3 = cv2.split(frame)


white = np.zeros(t1.shape, np.uint8)
white[:] = 255 # // !	

gray = np.zeros(t1.shape, np.uint8)
gray[:] = 128

black = np.zeros(t1.shape, np.uint8)
black[:] = 255 # // !
#black[:] = 0

while Preview:
	a=a+1
	#check,frame = video.read()
	
	#print(check,frame)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	h,s,v = cv2.split(hsv)
	#cv2.imshow('Capturing', gray)
	#cv2.imshow('Capturing', frame)
	key=cv2.waitKey(1)
	Preview = (key != ord(' '))
	

	
	#upper bound is:
	l_b = np.array([0,  0,  0])
	u_b = np.array([255,0,255])
	mask = cv2.inRange(hsv, l_b, u_b)
	
	# IF:
	# Saturation is HIGH and VALUE is LOW
	# OR:
	# Saturation is VERY HIGH
	# OR:
	# VALUE is VERY LOW
	# THEN:
	# Pixel is GRAY
	
	m1 = cv2.inRange(s, 235, 255) # Sat is VERY HIGH
	m2 = cv2.inRange(s, 160, 255) # Sat is HIGH
	m3 = cv2.inRange(v, 0,   64 ) # Value is LOW
	m4 = cv2.inRange(v, 0,   45 ) # Value is VERY LOW
	
	# Sat is >= 70% and Val <= 60%
	m2 = cv2.inRange(s, 0.70*256, 255) # Sat is HIGH
	m3 = cv2.inRange(v, 0,   0.60*256 ) # Value is LOW
	
		
	m23 = cv2.bitwise_and(m2,m3)
	m123 = cv2.bitwise_or(m1,m23)
	m1234 = cv2.bitwise_or(m123,m4)
	m234 = cv2.bitwise_or(m23,m4)  # // !
	
	x1 = cv2.bitwise_and(gray,m23)

	
	# Pixel is WHITE:
	# Sat is >= 80% and Val >= 40%
	m7 = cv2.inRange(s, 0.80*256, 255) 
	m8 = cv2.inRange(v, 0.40*256, 255)
	m78 = cv2.bitwise_and(m7, m8)
	
	x4 = cv2.bitwise_or(x1, m78)

	# IF:
	# Saturation is LOW
	# Pixel is BLACK

	m5 = cv2.inRange(s, 0, 76) # Sat is LOW
	

	x2 = cv2.bitwise_and(black,m5)
	
	# ELSE:
	# Pixel is WHITE
	x3 = cv2.add(x1,x2)
	
	cv2.imshow('Capturing', x3)

a = np.array(gray)

mpimg.imsave('rgb.png', frame)
#mpimg.imsave('gray.png', a, cmap='gray')
mpimg.imsave('hsv.png', hsv)

video.release()
cv2.destroyAllWindows()
