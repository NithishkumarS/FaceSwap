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
		coords[i] = (shape.part(i).y, shape.part(i).x)
	# return the list of (y, x)-coordinates
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
		pt1 = (t[1],t[0])
		pt2 = (t[3],t[2])
		pt3 = (t[5],t[4])
		if checkPoint(pt1,rect) and checkPoint(pt2,rect) and checkPoint(pt3,rect):
			cv2.line(image,pt1,pt2,(255,255,255),1)
			cv2.line(image,pt2,pt3,(255,255,255),1)
			cv2.line(image,pt3,pt1,(255,255,255),1)
			modified_triangles.append([pt1,pt2,pt3])
			# Compute descriptor index of all the triangle vertices
			idx_1 = np.where((landmark_pts == (t[0],t[1])).all(axis=1))[0][0]
			idx_2 = np.where((landmark_pts == (t[2],t[3])).all(axis=1))[0][0]
			idx_3 = np.where((landmark_pts == (t[4],t[5])).all(axis=1))[0][0]
			descriptor_index.append([idx_1, idx_2, idx_3])
	return image, descriptor_index, modified_triangles

def drawModifiedTriangles(image,triangles):
	# print(len(triangles), len(triangles[0]))
	for t in triangles:	#check with zack
		pt1 = (t[0][1],t[0][0])
		pt2 = (t[1][1],t[1][0])
		pt3 = (t[2][1],t[2][0])
		cv2.line(image,pt1,pt2,(255,255,255),1)
		cv2.line(image,pt2,pt3,(255,255,255),1)
		cv2.line(image,pt3,pt1,(255,255,255),1)
	return image

def findPoints(image,triangles,mask = False):
	
	if len(np.shape(image))==3:
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image[:,:]=0
	print(len(triangles), len(triangles[0]))

	if len(triangles) > 3:
		for t in np.array(triangles,dtype=np.int):
			cv2.fillPoly(image,np.array([t]),255)
	else:
		triangles = np.array(triangles,dtype=np.int)
		print(triangles.shape)
		# print(triangles)
		cv2.fillPoly(image,np.array([triangles]),255)
	points = np.transpose(np.where(image==255))
	if mask:
		return(points)
	else:
		return(points)

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
			cv2.circle(image,(y,x),1, (0,0,255),-1)
			subdiv.insert((x,y))

		cv2.rectangle(image,pt1,pt2,(255,0,0))
		triangles = subdiv.getTriangleList()
		image,descriptor_index, triangles = drawTriangles(image,triangles,bound_rect, shape)
		face_triangles.append(triangles)
		face_descriptor_index.append(descriptor_index)
		face_points.append(findPoints(gray,triangles))
	return face_shapes,face_triangles,face_points,image, face_descriptor_index

def triangulation(source, target, mode):

	triangles_target = []
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('Descriptors/shape_predictor_68_face_landmarks.dat')
	source_shapes, source_triangles, source_points, source_anno, descriptor_index = findFeatures(copy.deepcopy(source), predictor, detector)
	target_shapes, _, target_points, _, _ = findFeatures(copy.deepcopy(target), predictor, detector)
	
	target_anno = copy.deepcopy(target)
	target_triangles = []

	if mode == 1:
		if len(source_shapes)>1:
			source_shapes=source_shapes[0]
			source_triangles=source_triangles[0]
			source_points = source_points[0]
			descriptor_index = descriptor_index[0]
		target_shapes = target_shapes[0]
	elif mode == 2:
		if len(source_shapes)<2:
			print('Not enough faces detected for this mode')
			return False
		source_shapes=source_shapes[0]
		source_triangles=source_triangles[0]
		source_points = source_points[0]
		descriptor_index = descriptor_index[0]
		target_shapes = target_shapes[1]
		target_points = target_points[1]
	for t in descriptor_index[0]:
		# print(t[0],t[1],t[2])
		target_triangles.append([target_shapes[t[0]],target_shapes[t[1]],target_shapes[t[2]]])
	print('target triangles', len(target_triangles))
	target_anno = drawModifiedTriangles(copy.deepcopy(target),target_triangles)
		# Output
	# cv2.imshow('Source Annotations', source_anno)
	# cv2.imshow('Target Annotations', target_anno)
	# cv2.imshow('target', target)
	# cv2.imshow('Original Image', source)
	# cv2.waitKey(0)
	# cv2.imwrite('face.jpg', anno)
	return source_triangles, target_triangles

def compute_barycentric(triangle):
	BC = np.zeros((3,3), dtype='float')
	for i, point in enumerate(triangle):
		# print(point)
		BC[0,i] = point[0]
		BC[1, i] = point[1]
		BC[2, i] = 1
	return BC

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

# def roi_triangles(feature_pts):
# 	# print('check:::',feature_pts[0])
# 	minx, maxx = np.min(np.array(feature_pts, dtype=np.int32)[:,0]), np.max(np.array(feature_pts, dtype=np.int32)[:,0]+1)
# 	miny, maxy = np.min(np.array(feature_pts, dtype=np.int32)[:, 1]), np.max(np.array(feature_pts, dtype=np.int32)[:, 1] + 1)
# 	# print(minx, maxx, miny, maxy)
# 	face_points = [(i,j,1) for i in range(minx, maxx) for j in range(miny, maxy)]
# 	return face_points

def checkGreek(greek):
	alpha = greek[0]
	beta = greek[1]
	gamma = greek[2]
	flag = False
	# print(alpha+beta+gamma)
	if 0<=alpha<=1 and 0<=beta<=1 and 0<=gamma<=1:
		if 0<=alpha+beta+gamma<=1.1:
			flag = True
	return flag

def interpolate(source, pts):
		'''
		:param source: Source image to be interpolated
		:param pts: Target image applied with inverse Affine transform
		:return: images values of one corresponding triangle
		'''
		rounded_pts = np.int32(pts)
		# print(np.array(pts).shape, rounded_pts.shape)
		dx, dy = pts - rounded_pts
		# print(pts[0], rounded_pts[0], dx)
		# print(pts[1], rounded_pts[1], dy)
		# print('diff', dx, dy)

		val00 = source[rounded_pts[1], rounded_pts[0]]
		val01 = source[rounded_pts[1], rounded_pts[0] + 1]
		val11 = source[rounded_pts[1] + 1, rounded_pts[0]]
		val10 = source[rounded_pts[1] + 1, rounded_pts[0] + 1]

		axis1 = val01.T * dx + val00.T * (1 - dx)
		axis2 = val10.T * dx + val11.T * (1 - dx)
		interp = axis1 *(1- dy) + axis2 * dy			#(1 - dy)
		# print(len(interp))
		# print()
		return interp
def swap_faces(source, target, source_triangles, target_triangles,flag=False):

    # Choosing points corresponding to
	points_b = findPoints(target,target_triangles,mask=True)
	new_image = copy.deepcopy(target)
	mask = np.zeros(target.shape,dtype=np.uint8)

	for ti in range(len(source_triangles)):
		Ta = source_triangles[ti]
		Tb = target_triangles[ti]
		A_delta = compute_barycentric(Ta)
		B_delta = compute_barycentric(Tb)
		if flag:
			ss
			for p in points_b:
				pnt = [p[1],p[0],1]
				A_delta = compute_barycentric(Ta)
				B_delta = compute_barycentric(Tb)
				if np.linalg.det(B_delta) == 0:
					B_delta[0,0] = B_delta[0,0]+1
				greek = np.matmul(np.linalg.inv(B_delta),pnt)

				if checkGreek(greek):
					ya,xa,za = np.dot(A_delta,greek)
					ya = np.int(ya/za)
					xa = np.int(xa/za)
					new_image[p[0],p[1]]=source[ya,xa]
					mask[p[0],p[1]] = [255,255,255]
					# new_image[p[0],p[1]]=[255,0,0]
					# new_image[ya,xa] = [0,0,255]
					# cv2.imshow('Progress',new_image)
					# cv2.waitKey(1)
		else:
			import pdb
			# pdb.set_trace()
			for p in points_b:
				pnt = [p[1],p[0],1]
				A_delta = compute_barycentric(Ta)
				B_delta = compute_barycentric(Tb)
				if np.linalg.det(B_delta) == 0:
					B_delta[0,0] = B_delta[0,0]+1
				greek = np.matmul(np.linalg.inv(B_delta),pnt)

				if checkGreek(greek):
					ya,xa,za = np.dot(A_delta,greek)
					ya = ya/za		#np.int(ya/za)
					xa = xa/za	#np.int(xa/za)
					new_image[p[1],p[0]]=interpolate(source, [ya,xa])
					# source[xa,ya]
					mask[p[1],p[0]] = [255,255,255]
	M = cv2.moments(mask[:,:,0])
	cX = int(M["m10"]/M["m00"])
	cY = int(M["m01"]/M["m00"])

	center = (cX,cY)
	output = cv2.seamlessClone(new_image, target,mask, center, cv2.NORMAL_CLONE)
	return output,new_image

def main():
	Parser = argparse.ArgumentParser()
	# Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')
	Parser.add_argument('--sourceImage', default='Data/Set1/face_100.jpg', help='Directory that contains images for panorama sticking')
	Parser.add_argument('--targetImage', default='Data/Set1/target.jfif', help='Directory that contains images for panorama sticking')
	Parser.add_argument('--mode', default='2', help='mode = 1:swap a face from a source image into a target image 2:swap face between two people in the same image')

	Args = Parser.parse_args()
	source_name = Args.sourceImage
	target_name = Args.targetImage
	mode = int(Args.mode)

    # for i in range(100):
	if mode == 2:
		target_name = source_name
		source = cv2.imread(source_name)
		target = cv2.imread(target_name)
		cv2.imshow('Original Image',source)
	if mode ==1:
		print('here', source_name)

		source = cv2.imread('Data/Set1/face_1.jpg')
		target = cv2.imread(target_name)
		# cv2.imshow('Target Image',target)
		# cv2.imshow('Source Image',source)

	# else:
	# 	print('This is not a valid mode')
	# 	return False
	# cv2.waitKey(0)
	source_triangles, target_triangles = triangulation(source, target, mode)
	# print('s',np.shape(source_triangles))
	# print('t',np.shape(target_triangles))

	output_img,new_image = swap_faces(source, target,source_triangles[0], target_triangles)
	if mode == 2:
		output_img,new_image = swap_faces(source, new_image, target_triangles, source_triangles,flag = True)
	cv2.imshow('Swapped',output_img)
	cv2.imshow("N0 Blending", new_image)
	cv2.waitKey(0)




if __name__=='__main__':
		main()


