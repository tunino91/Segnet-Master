import cv2
import numpy as np
import os
from sklearn import preprocessing
import reader
import imageProcess
import SimpleITK as sitk
import ImageRegistrationMethod1,ImageRegistrationMethod2,ImageRegistrationMethod3,ImageRegistrationMethod4,ImageRegistrationMethod5

if __name__ == '__main__':

	pathToEyeVideoDataset = '/Users/tunaozerdem/Documents/UCF MASTER /Independent Study/ReadVideoByFrame/EyeVideos'
	pathToMain,tail = os.path.split(pathToEyeVideoDataset)
	pathToRegistrationResults = pathToMain + '/' + 'RegistrationResults'
	pathToEyeFrames = pathToMain + '/' + 'EyeFrames'
	

	if (os.path.exists(pathToEyeFrames) == False):
		reader.prepareFolders(pathToEyeVideoDataset)
		reader.readVideosWiteFrames(pathToEyeVideoDataset)
		imageProcess.cropFrames(pathToEyeFrames)

	else:
		frameFoldersNames = reader.getFolderNames(pathToEyeFrames)
		for i in frameFoldersNames:
			pathToEachFrame = pathToEyeFrames + '/' + i
			pathToEachRegistrationVideoFolder = pathToRegistrationResults + '/' + i
			frameNames = reader.getFileNames(pathToEachFrame)
			fixed_img_path = pathToEachFrame + '/' + frameNames[int(len(frameNames)/2)]
			
			for i in frameNames:
				fullpathToSingleFrame = pathToEachFrame + '/' + i
				fullpathToWriteRegistered = pathToEachRegistrationVideoFolder + '/' + i
				moving_img_path = fullpathToSingleFrame 
				registeredImagePath = fullpathToWriteRegistered
				ImageRegistrationMethod1.execute(fixed_img_path,moving_img_path,registeredImagePath)






	



	