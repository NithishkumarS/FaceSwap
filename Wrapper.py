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
	h,w,_ = np.shape(image)
	modified_triangles = []
	descriptor_index = []
	triangles_cods = []
	for t in triangles:
		pt1 = (t[0],t[1])
		pt2 = (t[2],t[3])
		pt3 = (t[4],t[5])
		if t[0]>0 and t[0]<h and t[1]>0 and t[1]<w and t[2]>0 and t[2]<h and t[3]>0 and t[3]<w and t[4]>0 and t[4]<h and t[5]>0 and t[5]<w:
			# print(landmark_pts)
			# print('points',pt1,pt2,pt3)
			cv2.line(image, pt1, pt2, (255, 255, 255), 1)
			cv2.line(image, pt2, pt3, (255, 255, 255), 1)
			cv2.line(image, pt3, pt1, (255, 255, 255), 1)
			pts = [pt1, pt2, pt3]
			triangles_cods.append(pts)
			# Compute descriptor index of all the triangle vertices
			idx_1 = np.where((landmark_pts == pt1).all(axis=1))[0][0]
			idx_2 = np.where((landmark_pts == pt2).all(axis=1))[0][0]
			idx_3 = np.where((landmark_pts == pt3).all(axis=1))[0][0]
			descriptor_index.append([idx_1, idx_2, idx_3])
	return image, descriptor_index, triangles_cods

def findPoints(image,triangles):
	if len(np.shape(image))==3:
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = image + 1
	# print(len(triangles))
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

def findFeatures(image, predictor, detector):
	face_points = []
	face_triangles = []
	face_shapes = []
	face_descriptor_index = []
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
		image,descriptor_index, triangles = drawTriangles(image,triangles,bound_rect, shape)
		face_triangles.append(triangles)
		face_descriptor_index.append(descriptor_index)
		# face_points.append(findPoints(gray,triangles))
	return face_shapes,face_triangles,face_points,image, face_descriptor_index

def triangulation(source, target):

	triangles_target = []
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('Descriptors/shape_predictor_68_face_landmarks.dat')
	souce_shapes, source_triangles, source_points, source_anno, descriptor_index = findFeatures(copy.deepcopy(source), predictor, detector)
	target_shapes, _, target_points, _, _ = findFeatures(copy.deepcopy(source), predictor, detector)
	
	target_anno = copy.deepcopy(target)
	target_triangles = []
	print(np.shape(descriptor_index))
	target_shapes = target_shapes[0]
	for t in descriptor_index[0]:
		target_triangles.append([target_shapes[t[0]],target_shapes[t[1]],target_shapes[t[2]]])

		# Output
	# cv2.imshow('Annotations', anno)
	cv2.imshow('target', target)
	cv2.imshow('Original Image', source)
	# cv2.waitKey(0)
	# cv2.imwrite('face.jpg', anno)
	return source_triangles, target_triangles

def compute_barycentric(triangle):
	BC = np.zeros((3,3), dtype='float')
	for i, point in enumerate(triangle):
		print(i)
		BC[0,i] = point[0]
		BC[1, i] = point[1]
		BC[2, i] = 1
	return BC

def f(source_triangle, target_triangle, pt):
	AC = compute_barycentric(st)
	# import  pdb
	# pdb.set_trace()
	BC = compute_barycentric(tt)
	pt = [pt[0], pt[1], 1]
	greek = np.matmul(np.linalg.inv(BC), np.array(pt))
	# print('s', greek)
	# print('sum', np.sum(greek))
	X = np.matmul(AC, greek)
	return X

def compute_affine(source_triangles, target_triangles):
	'''
	:param source_triangles: delauny triangles in source image
	:param target_triangles: delauny triangles in target image
	:return: affine transform matrix list
	'''

	# print(len(source_triangles), len(target_triangles))
	for i in range(len(source_triangles)):
		source_mat = compute_barycentric(source_triangles[i])
		target_mat = compute_barycentric(target_triangles[i])
		mat = np.dot(source_mat ,np.linalg.inv(target_mat))
		yield mat, np.linalg.inv(target_mat)

def roi_triangles(feature_pts):
	# print('check:::',feature_pts[0])
	minx, maxx = np.min(np.array(feature_pts, dtype=np.int32)[:,0]), np.max(np.array(feature_pts, dtype=np.int32)[:,0]+1)
	miny, maxy = np.min(np.array(feature_pts, dtype=np.int32)[:, 1]), np.max(np.array(feature_pts, dtype=np.int32)[:, 1] + 1)
	# print(minx, maxx, miny, maxy)
	face_points = [(i,j,1) for i in range(minx, maxx) for j in range(miny, maxy)]
	return face_points

def swap_faces(source, target, transformation_matrices,B, source_triangles, target_triangles):
	# roi_pts = np.array( roi_triangles(target_triangles))
	# print(roi_pts.shape)
	# roi_index = np.ones_like(roi_pts[:,0]) * np.inf

    # Choosing points corresponding to
	for ti in range(len(source_triangles)):
		Ta = source_triangles[ti]
		Tb = target_triangles[ti]
		points_b = findPoints
		# print(roi_pts.T)
		# print(B[triangle_index])
		greek = np.dot(B[triangle_index], roi_pts.T )
		# print(greek.shape)
		# print('sum size',len(np.sum(greek, axis = 0)))
		# for i in np.sum(greek, axis = 0):
		# 	print(i)
		# print(greek)
		# print('val:',np.where(np.sum(greek, axis=0)[0] <= 1.0))
		# print('no of elements', triangle_index, ' : ',len(np.where(np.sum(greek, axis=0)[0] <= 1)) )
		roi_index[np.where(np.sum(greek, axis=0) <= 1)] = triangle_index
		# print(np.where(roi_index != 0))
	transformed_image = np.zeros_like(source)
	for i in range(len(source_triangles)):
		pts = roi_pts[roi_index == i]
		transformed_coods = np.dot(transformation_matrices[i], pts.T )
		(x,y) = pts
		transformed_image[y,x] = interpolation(source, transformed_coods)

	return transformed_image

def main():
	source = cv2.imread('Data/2faces.jpeg')  # 2faces.jpeg')
	target = cv2.imread('Data/2faces.jpeg')  # 2faces.jpeg')
	print(source.shape)
	source_triangles, target_triangles = triangulation(source, target)
	# cv2.waitKey(0)
	# Compute Affine transformation matrices
	ret = np.array(list(compute_affine( target_triangles, source_triangles[0])))
	print(np.shape(ret))
	print(np.shape(source_triangles[0]))
	# ret = list(compute_affine(source_triangles[0], target_triangles))[:,0]
	transformation_matrices = ret[:,0]
	B = np.array(ret[:,1])
	output_img = swap_faces(source.copy(), target.copy(), transformation_matrices, B, source_triangles[0], target_triangles)

	# for tt,st in zip(target_triangles, source_triangles):
	# 	# AC = compute_barycentric(st)
	# 	# # import  pdb
	# 	# # pdb.set_trace()
	# 	# BC = compute_barycentric(tt)
	# 	# pt = [tt[0][0], tt[0][1], 1]
	# 	# greek = np.matmul(np.linalg.inv(BC), np.array(pt))
	# 	# print('s', greek)
	# 	# print('sum', np.sum(greek))
	# 	# X = np.matmul(AC, greek)
	#
	# 	X = f(st, tt, pt)
    #     x = X[0] / X[2]
	#     y = X[1] / X[2]



if __name__=='__main__':
		main()


