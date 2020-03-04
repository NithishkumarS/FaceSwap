'''

References:
	https://pysource.com/2019/05/03/matching-the-two-faces-triangulation-face-swapping-opencv-with-python-part-3/
	https://www.learnopencv.com/delaunay-triangulation-and-voronoi-diagram-using-opencv-c-python/


'''

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

def drawTriangles(image,triangles,rect,landmark_pts,fill=1):
	'''
	:param image:
	:param triangles: Intial triangle points from descriptors
	:param rect:
	:param landmark_pts: Feature descirptor points of face 1
	:return:
	'''
	modified_triangles = []
	descriptor_index = []
	for t in triangles:
		pt1 = (t[0],t[1])
		pt2 = (t[2],t[3])
		pt3 = (t[4],t[5])
		cv2.line(image, pt1, pt2, (255, 255, 255), 1)
		cv2.line(image, pt2, pt3, (255, 255, 255), 1)
		cv2.line(image, pt3, pt1, (255, 255, 255), 1)

		# Compute descriptor index of all the triangle vertices
		idx_1 = np.where((landmark_pts == pt1).all(axis=1))
		idx_2 = np.where((landmark_pts == pt2).all(axis=1))
		idx_3 = np.where((landmark_pts == pt3).all(axis=1))
		descriptor_index.append([idx_1, idx_2, idx_3])
	return image, descriptor_index

def findPoints(image,triangles):
	if len(np.shape(image))==3:
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = image + 1
	print(len(triangles))
	for t in np.array(triangles,dtype=np.int):
		cv2.fillPoly(image,np.array([t]),0)
	points = np.transpose(np.where(image==0))
	return points

def calcU(point1,point2):
	x1 = point1[0]
	y1 = point1[1]
	x2 = point2[0]
	y2 = point2[1]
	r_2 = (np.linalg.norm([x1,y1]-[x2-y2]))**2
	U = r_2 * np.log(r_2)
	return U

def calcK(points):
	K = np.zeros((len(points1,len(points2))),dtype=np.float32)
	for i,pointsi in enumerate(points):
		for j,pointsj in enumerate(points):
			K[i,j] = calcU(pointi,pointj)
	return K

def getP(pointsi):
	ones = np.zeros((len(pointsi),1),np.float32) + 1;
	P = np.hstack((pointsi,ones))
	Pt = np.transpose(P)
	return P,Pt

def getV(points):
	V = np.hstack((points,0,0,0))
	return V

def getSpline(points1,points2):
	lamda = 0.0001
	identity = lamda * np.identity(len(points1)+3,dtype=np.float32)
	zeros = np.zeros((3,3))
	K = calcK(points1)
	P, Pt = getP(points1)
	A = np.linalg.inv(np.vstack((np.hstack((K,P)), np.hstack((Pt,zeros))))+identity)
	Vx = getV(points2[:,0])
	Vy = getV(points2[:,1])
	valuesx = np.matmul(A,Vx)
	valuesy = np.matmul(A,Vy)

def calcSpline1d(point,values, points):
	x = point[1]
	y = point[0]
	a1 = values[-1]
	ay = values[-2]
	ax = values[-3]
	w = values[:-3]
	U = []
	for pnti in points:
		U.append(calcU(pnti,point))
	f = a1+ax*x+ay*y+np.sum(w*U)
	return f

def calcSpline(point, valuesx, valuesy, points):
	fx = calcSpline1d(valuesx,point)
	fy = calcspline1d(valuesy,point)
	return fx,fy

def findFeatures(image):
	face_points = []
	face_triangles = []
	face_shapes = []
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
		face_shapes.append(shape)
		subdiv = cv2.Subdiv2D(bound_rect)
		for (i,(x,y)) in enumerate(shape):
			cv2.circle(image,(x,y),1, (0,0,255),-1)
			subdiv.insert((x,y))
		cv2.rectangle(image,pt1,pt2,(255,0,0))
		triangles = subdiv.getTriangleList()
		print('No of triangles::',len(triangles))
		image,descriptor_index = drawTriangles(image,triangles,bound_rect, shape)
		face_triangles.append(triangles)
		face_points.append(findPoints(gray,triangles))
	return face_shapes,face_triangles,face_points,image



def main():
	source = cv2.imread('Data/Set1/face_1.jpg')  # 2faces.jpeg')
	target = cv2.imread('Data/Set1/target.jpg')  # 2faces.jpeg')
	print(target.shape)
	shapes, triangles, points, anno = findFeatures(copy.deepcopy(source))



	# Output
	cv2.imshow('Annotations', anno)
	cv2.imshow('Origonal Image', image)
	# cv2.resizeWindow('face', 1600, 1200)
	cv2.waitKey(0)
	cv2.imwrite('face.jpg', anno)

if __name__=='__main__':
		main()


