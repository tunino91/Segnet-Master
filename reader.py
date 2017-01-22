import cv2
import numpy as np
import os
import glob
import subprocess


### Use it if u need the only folder names inside a folder ###
## returns a list of names of the content of the folder ##
def getFolderNames (folder):

	folderContent = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]
	
	if not folderContent:
		return False

	else:
		return folderContent


### Use it if u need only the file names inside a folder ###
## returns a list of names of the content of the folder ##
def getFileNames (folder):
	folderContent = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
	s = set(folderContent)
	if '.DS_Store' in s:
		folderContent.remove('.DS_Store')
	
	if not folderContent:
		return False

	else:
		return folderContent



def prepareFolders(pathToEyeVideosFolder): # EyeVideosFolder : /Users/tunaozerdem/Documents/UCF MASTER /Independent Study/ReadVideoByFrame/EyeVideos
	videoFileNames = getFileNames(pathToEyeVideosFolder)
	pathToFramesFolder,tail = os.path.split(pathToEyeVideosFolder) # pathToFramesFolder: /Users/tunaozerdem/Documents/UCF MASTER /Independent Study/ReadVideoByFrame
	pathToFramesFolder = pathToFramesFolder + '/EyeFrames'
	pathToRegistrationResultsFolder = pathToFramesFolder + '/RegistrationResults'


	if (os.path.exists(pathToFramesFolder) == False):
		print 'Creating Folders To Save Frames For Each Video'
		os.mkdir(pathToFramesFolder)
		os.mkdir(pathToRegistrationResultsFolder)
		## For each videofile it reads, create a dedicated folder
		##  in Eyeframes for that videofile to put the frames in. 
		for i in videoFileNames:
			
			frameFolderPathForEachVideo = pathToFramesFolder + '/' + os.path.splitext(i)[0]
			registrationResultForEachVideoFolder = pathToRegistrationResultsFolder + '/' + os.path.splitext(i)[0]
			os.mkdir(frameFolderPathForEachVideo)
			os.mkdir(registrationResultForEachVideoFolder)

		
def readVideosWiteFrames(pathToEyeVideoDataset):
	videoFileNames = getFileNames(pathToEyeVideoDataset)
	pathToFramesFolder,tail = os.path.split(pathToEyeVideoDataset) # pathToFramesFolder: /Users/tunaozerdem/Documents/UCF MASTER /Independent Study/ReadVideoByFrame/EyeFrames
	pathToFramesFolder = pathToFramesFolder + '/' + 'EyeFrames'
	for i in videoFileNames:
		fullPathToASingleVideoFile = pathToEyeVideoDataset + '/' + i
		pathToFramesFolder = pathToFramesFolder + '/' + os.path.splitext(i)[0]
		subprocess.call(['ffmpeg', '-i', fullPathToASingleVideoFile, (pathToFramesFolder + '/' + os.path.splitext(i)[0] + '%03d' + '.jpg')])
		pathToFramesFolder,tail = os.path.split(pathToFramesFolder)


# def readVideo(fullPathToASingleVideoFile):
	
# 	#fullPathToASingleVideoFile: /Users/tunaozerdem/Documents/UCF MASTER /Independent Study/ReadVideoByFrame/EyeVideos/video_1.avi
# 	pathToEyeVideos,nameOfTheVideoFile = os.path.split(fullPathToASingleVideoFile)
# 	pathToFramesFolder = os.path.split(pathToEyeVideos) # pathToFramesFolder: /Users/tunaozerdem/Documents/UCF MASTER /Independent Study/ReadVideoByFrame/EyeFrames
# 	pathToFramesFolder = pathToFramesFolder + '/EyeFrames'

# 	if (os.path.exists(pathToFramesFolder) == False):
# 		print 'Creating Folders To Save Frames For Each Video'
# 		os.mkdir(pathToFramesFolder)
# 		counter = 0
# 		for i in namesOfTheVideoFiles:
# 			os.mkdir(pathsToSaveFramesForEachVideoFile[counter])
# 			subprocess.call(['ffmpeg', '-i', fullPathToASingleVideoFile, (pathsToSaveFramesForEachVideoFile[counter] + '/' + nameOfTheVideoFile + '%03d' + '.jpg')])

	
# 	elif(os.path.exists(pathToFramesFolder) == True):
# 		if(os.path.exists(pathsToSaveFramesForEachVideoFile) == False):
# 			if(os.path.exists(pathToSaveFramesForEachClass) == False):
# 				os.mkdir(pathToSaveFramesForEachClass)
# 			os.mkdir(pathToSaveFramesForEachVideoFile)
# 			subprocess.call(['ffmpeg', '-i', fullPathToASingleVideoFile, ( pathToSaveFramesForEachVideoFile + '/' + nameOfTheVideoFile + '%03d' + '.jpg')])
# 		elif(os.path.exists(pathToSaveFramesForEachVideoFile) == True):
# 			subprocess.call(['ffmpeg', '-i', fullPathToASingleVideoFile, ( pathToSaveFramesForEachVideoFile + '/' + nameOfTheVideoFile + '%03d' + '.jpg')])
	

	 

	




