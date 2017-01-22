import cv2
import numpy as np
import os
from sklearn import preprocessing
import reader
import matplotlib as plt

def cropFrames(pathToEyeFrames):
	framesFolderNames = reader.getFolderNames(pathToEyeFrames)
	for i in framesFolderNames:	
		pathToOneFrameFolder = pathToEyeFrames + '/' + i

		for nameOfTheFrame in reader.getFileNames(pathToOneFrameFolder):
			fullPathToOneFrame = pathToOneFrameFolder + '/' + nameOfTheFrame
			#print 'Full path to one frame:', fullPathToOneFrame
			img = cv2.imread(fullPathToOneFrame)
			crop_img = img[0:80, 0:60] # Crop from x, y, w, h -> 100, 200, 300, 400
			#plt.imshow(crop_img)
			cv2.imwrite(fullPathToOneFrame,crop_img)