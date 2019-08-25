import time
import PIL
import os
import math
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import tkinter as tk
from tkinter import Label,Tk
import warnings
num = 0
warnings.simplefilter('ignore', np.RankWarning)
plt.ion()
plt.plot()
p = None
def getFunction(points,xmax,xmin,ymax):
	try:
		global p
		x = ([data[1] for data in points])
		y = ([260-data[0] for data in points])
		z = np.polyfit(x, y, 15) #Determines Best Fit Polynomial
		p = np.poly1d(z) #Converts Best Fit Polynomial to Readable Function
		p_int = np.polyint(p) #Integrates Polynomial
		area = int(p_int(xmax)-p_int(xmin)) #Fundamental Theorem of Calculus II
		multiplier = (9.866*8.5)/(348*260)
		area_inches = round(area*multiplier,2)
		print("\n\n\n\n\n\n\n\n\nf(x) = "+p+"\n∫f(x) = "+p_int+"\nArea: "+str(area)+" pixels or "+str(area_inches)+" square inches")
		plt.figure(1)
		plt.suptitle("Area: "+str(area_inches)+" inches²", fontsize=20)
		xp = np.linspace(0, 348, 348)
		_ = plt.plot([0],[348],'.',xp[xmin:xmax], p(xp[xmin:xmax]), '-')
		plt.fill_between(xp[xmin:xmax], p(xp[xmin:xmax]),0, facecolor='blue', alpha=0.1)
		plt.xlim(0,348)
		plt.ylim(0,260)
		plt.draw()
		plt.pause(0.0001)
		plt.savefig('foo.png')
		plt.clf()
	except Exception as e:
		error = True

def clearFunction():
	plt.ylim(0,260)
	plt.xlim(0,348)
	plt.draw()
	plt.pause(0.0001)
	plt.clf()
clearFunction()

def dotproduct(v1, v2):
	return sum((a*b) for a, b in zip(v1, v2))

def length(v):
	return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
	return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

def skeletonize(img):
	size = np.size(img)
	skel = np.zeros(img.shape,np.uint8)
	 
	ret,img = cv2.threshold(img,127,255,0)
	element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
	done = False
	 
	while(not done):
		eroded = cv2.erode(img,element)
		temp = cv2.dilate(eroded,element)
		temp = cv2.subtract(img,temp)
		skel = cv2.bitwise_or(skel,temp)
		img = eroded.copy()
	 
		zeros = size - cv2.countNonZero(img)
		if zeros==size:
			done = True
	return skel

def draw_lines(img,lines):
	newlines = []
	slopes = []
	if lines is not None:
		for line in lines:
			x1 = line[0][0]
			x2 = line[0][2]
			y1 = line[0][1]
			y2 = line[0][3]
			slope = (y2-y1)/(x2-x1)
			yintercept = y2-(slope*x2)
			magnitude = math.sqrt(math.pow(x2-x1,2)+math.pow(y2-y1,2))
			vector = ((x2-x1)/magnitude,(y2-y1)/magnitude)
			coords = line[0]
			scale1 = -100
			scale2 = 100
			keepline = True
			for i in range(0,len(slopes)):
				if keepline == True:
					if (abs(yintercept+0.0001)/abs(slopes[i][1])+0.0001) > 1.05 or (abs(yintercept+0.0001)/abs(slopes[i][1])+0.0001) < 0.95:
						keepline = True
					else:
						keepline = False
			slopes.append([slope,yintercept])
			if keepline == True:
				newlines.append([[int(vector[0]*scale1)+x1,int(vector[1]*scale1)+y1,int(vector[0]*scale2)+x2,int(vector[1]*scale2)+y2]])
	compare_slopes(img,lines)
def compare_slopes(img,lines):
	if lines is not None:
		for line in lines:
			x1a = line[0][0]
			x2a = line[0][2]
			y1a = line[0][1]
			y2a = line[0][3]
			slopea = (y2a-y1a)/(x2a-x1a)
			yintercepta = y2a-(slopea*x2a)
			vector = ((x2a-x1a),(y2a-y1a))
			for testline in lines:
				x1b = testline[0][0]
				x2b = testline[0][2]
				y1b = testline[0][1]
				y2b = testline[0][3]
				testvector = ((x2b-x1b),(y2b-y1b))
				slopeb = (y2b-y1b)/(x2b-x1b)
				yinterceptb = y2b-(slopeb*x2b)
				intersect_x = (yinterceptb-yintercepta)/(slopea-slopeb)
				intersect_y = slopea*intersect_x + yintercepta
				if np.isnan(intersect_x) == False and np.isinf(intersect_x) == False and np.isinf(intersect_y) == False:
					cv2.circle(img,(int(intersect_x),int(intersect_y)), 3, (0,0,255), -1)
def CaptureImage():
	cap = cv2.VideoCapture(0) #Choose proper webcam input
	while(True):
		ret, frame = cap.read()
		oldframe = frame
		frame = frame
		function = oldframe
		hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
		lower_green = np.array([32,0,0])
		upper_green = np.array([80,255,255])
		mask = cv2.inRange(hsv,lower_green,upper_green)
		frame = cv2.bitwise_and(frame, frame, mask = mask)
		frame = cv2.GaussianBlur(frame,(5,5),0)
		edges = cv2.Canny(frame,300,300)
		lines = cv2.HoughLinesP(edges, 1, np.pi/180, 25, np.array([0,0,0]), 25,100)
		kernel = np.ones((5,5),np.uint8)
		edges = cv2.dilate(edges,kernel,iterations = 1)
		cnt,contours = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
		contourFound = False
		functionDetected = False
		for contour in cnt:
			area = cv2.contourArea(contour)
			if area>10000 and contourFound == False:
				contourFound = True
				(x,y,w,h) = cv2.boundingRect(contour)
				hull = cv2.convexHull(contour)
				maxdistance1 = 0
				maxpoints1 = None
				maxpointindex1 = 0
				maxdistance2 = 0
				maxpoints2 = None
				for i,point1 in enumerate(hull):
					x1 = point1[0][0]
					y1 = point1[0][1]
					for point2 in hull:
						x2 = point2[0][0]
						y2 = point2[0][1]
						distance = math.sqrt(math.pow(x2-x1,2)+math.pow(y2-y1,2))
						if distance>maxdistance1:
							maxdistance1 = distance
							maxpoints1 = [(x1,y1),(x2,y2)]
							maxpointindex1 = i
				for i,point1 in enumerate(hull):
					x1 = point1[0][0]
					y1 = point1[0][1]
					for k,point2 in enumerate(hull):
						x2 = point2[0][0]
						y2 = point2[0][1]
						distance = math.sqrt(math.pow(x2-x1,2)+math.pow(y2-y1,2))
						if distance>maxdistance2 and i != maxpointindex1 and k != maxpointindex1:
							vectorangle = angle([maxpoints1[0][0],maxpoints1[0][1],maxpoints1[1][0],maxpoints1[1][1]],[x1,y1,x2,y2])
							vectorangle = 180*vectorangle/np.pi
							d1 = math.sqrt(math.pow(x2-maxpoints1[0][0],2)+math.pow(y2-maxpoints1[0][1],2))
							d2 = math.sqrt(math.pow(x2-maxpoints1[1][0],2)+math.pow(y2-maxpoints1[1][1],2))
							d3 = math.sqrt(math.pow(x1-maxpoints1[0][0],2)+math.pow(y1-maxpoints1[0][1],2))
							d4 = math.sqrt(math.pow(x1-maxpoints1[1][0],2)+math.pow(y1-maxpoints1[1][1],2))
							if vectorangle is not None:
								if vectorangle>22 and vectorangle<35:
									if d1/distance<0.1 or d2/distance<0.1 or d3/distance<0.1 or d4/distance<0.1:
										close = True
									else:
										maxdistance2 = distance
										maxpoints2 = [(x1,y1),(x2,y2)]
				if maxpoints1 is not None:
					x1 = maxpoints1[0][0]
					x2 = maxpoints1[1][0]
					y1 = maxpoints1[0][1]
					y2 = maxpoints1[1][1]
				if maxpoints2 is not None:
					x3 = maxpoints2[0][0]
					x4 = maxpoints2[1][0]
					y3 = maxpoints2[0][1]
					y4 = maxpoints2[1][1]
				functionDetected = True
				points = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
				points.sort(key=lambda x: x[0])
				left_points = [points[0],points[1]]
				right_points = [points[2],points[3]]
				left_points.sort(key=lambda x: x[1])
				right_points.sort(key=lambda x: x[1])
				x1 = left_points[0][0]
				y1 = left_points[0][1]
				x3 = left_points[1][0]
				y3 = left_points[1][1]
				x2 = right_points[0][0]
				y2 = right_points[0][1]
				x4 = right_points[1][0]
				y4 = right_points[1][1]
				cv2.circle(frame, (x1,y1), 5, (255,0,0))
				cv2.circle(frame, (x2,y2), 5, (255,0,0))
				cv2.circle(frame, (x3,y3), 5, (0,255,255))
				cv2.circle(frame, (x4,y4), 5, (0,255,255))
				pts1 = np.float32([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
				pts2 = np.float32([[0,0],[388,0],[0,300],[388,300]])
				M = cv2.getPerspectiveTransform(pts1,pts2)
				function = cv2.warpPerspective(function,M,(388,300))
				kernel = np.ones((5,5),np.uint8)
				function = function[20:280,20:368]
				function = cv2.cvtColor(function, cv2.COLOR_BGR2GRAY)
				function = cv2.Canny(function,200,200)
				function = cv2.dilate(function,kernel,iterations=2)
				function = cv2.erode(function,kernel,iterations=2)
				function = skeletonize(function)
				points = np.argwhere(function == 255)
				try:
					xmax = max([data[1] for data in points])
					xmin = min([data[1] for data in points])
					ymax = max([data[0] for data in points])
				except:
					xmax = 388
					xmin = 0
					ymax = 300
				getFunction(points,xmax,xmin,ymax)
				graph = cv2.imread('foo.png')
				graph = cv2.resize(graph, (oldframe.shape[1],oldframe.shape[0]), interpolation = cv2.INTER_AREA)
				width = graph.shape[1]
				height = graph.shape[0]
				margin = 60
				pts1 = np.float32([[margin,margin],[width-margin,margin],[margin,height-margin],[width-margin,height-margin]])
				pts2 = np.float32([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
				M = cv2.getPerspectiveTransform(pts1,pts2)
				graph = cv2.warpPerspective(graph,M,(oldframe.shape[1],oldframe.shape[0]))
				width = graph.shape[0]
				height = graph.shape[1]
				beta = (1.0 - 0.5)
				np.all(graph == [0, 0, 0], axis=2)
				graph[np.all(graph == [0, 0, 0], axis=2)] = oldframe[np.all(graph == [0, 0, 0], axis=2)]
		function2 = cv2.resize(function, (640,480), interpolation = cv2.INTER_AREA)
		cv2.imshow('frame',frame)
		cv2.imshow('function',function2)
		cv2.imshow('oldframe',oldframe)
		try:
			graph2 = cv2.resize(graph, (1280,960), interpolation = cv2.INTER_AREA)
			cv2.imshow('graph',graph2)
		except:
			oldframe2 = cv2.resize(oldframe, (1280,960), interpolation = cv2.INTER_AREA)
			cv2.imshow('graph',oldframe2)
			found = False
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
CaptureImage()