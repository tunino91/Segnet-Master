# Segnet-Master
This repository is dedicated for implementation of SegNet in Keras for Eye Vessel Segmentation. Each video file read, belongs to one patient in the dataset. Registration is the first step necessary towards preprocessing of the dataset. 

## Registration

All functions are called from main.py 

### Prerequisites
• SimpleITK --> Homebrew installation, on command line type:  < brew install SimpleITK > 
• OpenCV    --> http://www.pyimagesearch.com/2016/12/19/install-opencv-3-on-macos-with-homebrew-the-easy-way/

### Prepare DataSet Folders as Depicted Below:
Each video file read, belongs to one patient.
<img width="405" alt="screen shot 2017-01-21 at 10 22 48 pm" src="https://cloud.githubusercontent.com/assets/19553239/22179650/4115fe8c-e028-11e6-9748-8172b629237c.png">

    
### How to Run Registration:
1) On command line, switch to your downloaded folder where all .py files are present.

2) type: >> python main.py     

3) All of your registered images are stored in RegistrationResults folder which is created automatically by the script.

### Brief Description of the Algorithm:
First, the video files are read frame by frame and each frame is registered with the middle frame.

Transformation is rigid since the structure of the eye vessels will not change from one frame to another.


### Used Methods 
• Transformation Type: Rigid

• Interpolation: Cubic

• Similarity Function: Normalized Cross-Corrolation

• Optimizer: Will be determined..



