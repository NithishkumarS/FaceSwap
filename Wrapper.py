'''

References:
	https://pysource.com/2019/05/03/matching-the-two-faces-triangulation-face-swapping-opencv-with-python-part-3/
	https://www.learnopencv.com/delaunay-triangulation-and-voronoi-diagram-using-opencv-c-python/


'''

import argparse
import numpy as np
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
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
			modified_triangles.append([pt1,pt2,pt3])
		else:
			cv2.line(image, pt1, pt2, (255, 0, 0), 1)
			cv2.line(image, pt2, pt3, (255, 0, 0), 1)
			cv2.line(image, pt3, pt1, (255, 0, 0), 1)
			modified_triangles.append([pt1, pt2, pt3])

	return image,modified_triangles

def calcU(point1,point2):
	x1 = point1[0]
	y1 = point1[1]
	x2 = point2[0]
	y2 = point2[1]
	r_2 = (np.linalg.norm([x1,y1]-[x2-y2]))**2
	U = r_2 * np.log(r_2)
	return U

def calcK(points1,points2):
	K = np.zeros((len(points1,len(points2))),dtype=np.float32)
	for i,pointi in enumerate(points1):
		for j,poinj in enumerate(points2):
			K[i,j] = calcU(pointi,pointj)
	return K

def getP(pointsi):
	ones = np.zeros((len(pointsi),1),np.float32) + 1;
	P = np.hstack((pointsi,ones))
	Pt = np.transpose(P)
	return P,Pt


def findFeatures(image):
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('Descriptors/shape_predictor_68_face_landmarks.dat')

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
		for (i,(x,y)) in enumerate(shape):
			cv2.circle(image,(x,y),1, (0,0,255),-1)
			subdiv.insert((x,y))
		cv2.rectangle(image,pt1,pt2,(255,0,0))
		triangles = subdiv.getTriangleList()
		print('No of triangles::',len(triangles))
		image,triangles = drawTriangles(image,triangles,bound_rect)
	print(np.shape(triangles))
	cv2.imshow('face',image)
	cv2.imwrite('face.jpg', image)
	# cv2.resizeWindow('face', 1600, 1200)

	cv2.waitKey(0)
	return shape,triangles



def main():
	source = cv2.imread('Data/Set1/face_1.jpg')  # 2faces.jpeg')
	target = cv2.imread('Data/Set1/target.jpg')  # 2faces.jpeg')
	print(target.shape)
	# target = cv2.resize(target, (640, 480))
	# cv2.imshow('target', target)
	# cv2.waitKey(0)
	shape, triangles = findFeatures(source)


if __name__=='__main__':
		main()


# Face Warping using Thin Plate Spline


