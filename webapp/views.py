from django.shortcuts import render,redirect
from django.http import HttpResponse
import numpy as np
import cv2
import pickle
import streamlit as st
from PIL import Image
import os
import pickle


def run(request):
		userId = request.POST['userId']

		# Detect face
		#Creating a cascade image classifier
		faceDetect = cv2.CascadeClassifier(BASE_DIR+'/ml/haarcascade_frontalface_default.xml')
		#camture images from the webcam and process and detect the face
		# takes video capture id, for webcam most of the time its 0.
		cam = cv2.VideoCapture(0)

		# Our identifier
		# We will put the id here and we will store the id with a face, so that later we can identify whose face it is
		id = userId
		# Our dataset naming counter
		sampleNum = 0
		# Capturing the faces one by one and detect the faces and showing it on the window
		while(True):
			# Capturing the image
			#cam.read will return the status variable and the captured colored image
			ret, img = cam.read()
			#the returned img is a colored image but for the classifier to work we need a greyscale image
			#to convert
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			#To store the faces
			#This will detect all the images in the current frame, and it will return the coordinates of the faces
			#Takes in image and some other parameter for accurate result
			faces = faceDetect.detectMultiScale(gray, 1.3, 5)
			#In above 'faces' variable there can be multiple faces so we have to get each and every face and draw a rectangle around it.
			for(x,y,w,h) in faces:
				# Whenever the program captures the face, we will write that is a folder
				# Before capturing the face, we need to tell the script whose face it is
				# For that we will need an identifier, here we call it id
				# So now we captured a face, we need to write it in a file
				sampleNum = sampleNum+1
				# Saving the image dataset, but only the face part, cropping the rest
				cv2.imwrite(BASE_DIR+'/ml/dataset/user.'+str(id)+'.'+str(sampleNum)+'.jpg', gray[y:y+h,x:x+w])
				# @params the initial point of the rectangle will be x,y and
				# @params end point will be x+width and y+height
				# @params along with color of the rectangle
				# @params thickness of the rectangle
				cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,0), 2)
				# Before continuing to the next loop, I want to give it a little pause
				# waitKey of 100 millisecond
				cv2.waitKey(250)

			#Showing the image in another window
			#Creates a window with window name "Face" and with the image img
			cv2.imshow("Face",img)
			#Before closing it we need to give a wait command, otherwise the open cv wont work
			# @params with the millisecond of delay 1
			cv2.waitKey(1)
			#To get out of the loop
			if(sampleNum>35):
				break
		#releasing the cam
		cam.release()
		# destroying all the windows
		cv2.destroyAllWindows()

		return redirect('/')
