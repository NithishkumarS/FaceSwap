import argparse
import numpy as np
import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import os
import copy
import itertools
import random
import dlib
# from imutils import face_utils


def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

def rect2box(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right()
	h = rect.bottom()
	return (x,y),(w,h)

def checkPoint(pnt,rect):
	flag = False
	x = pnt[0]
	y = pnt[1]
	if x>rect[0] and y>rect[1] and x<rect[2] and y<rect[3]:
		flag = True
	return flag

def drawTriangles(image,triangles,rect):
	modified_triangles = []
	for t in triangles:
		pt1 = (t[0],t[1])
		pt2 = (t[2],t[3])
		pt3 = (t[4],t[5])
		if checkPoint(pt1,rect) and checkPoint(pt2,rect) and checkPoint(pt3,rect):
			cv2.line(image,pt1,pt2,(255,255,255),1)
			cv2.line(image,pt2,pt3,(255,255,255),1)
			cv2.line(image,pt3,pt1,(255,255,255),1)
			modified_triangles.append(t)
	return image,modified_triangles

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('Descriptors/shape_predictor_68_face_landmarks.dat')

image = cv2.imread('Data/2faces.jpeg')
size = image.shape
bound_rect = (0,0,size[1],size[0])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow('face',image)
# cv2.waitKey(0)

rects = detector(gray,1)
for (i, rect) in enumerate(rects):
	pt1,pt2 = rect2box(rect)
	shape = predictor(gray,rect)
	shape = shape_to_np(shape)
	subdiv = cv2.Subdiv2D(bound_rect)
	for (x,y) in shape:
		cv2.circle(image,(x,y),1, (0,0,255),-1)
		subdiv.insert((x,y))
	cv2.rectangle(image,pt1,pt2,(255,0,0))
	triangles = subdiv.getTriangleList()
	image,triangles = drawTriangles(image,triangles,bound_rect)
print(np.shape(triangles))
cv2.imshow('face',image)
cv2.waitKey(0)

