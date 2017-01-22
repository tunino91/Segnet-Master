# Segnet-Master
This repository is dedicated for implementation of SegNet in Keras for Eye Vessel Segmentation. Each video file read, belongs to one patient in the dataset. Registration is the first step necessary towards preprocessing of the dataset. 

## Registration
All functions are called from main.py 
### Prepare DataSet Folders:
    

First, the video files are read frame by frame and each frame is registered with the middle frame.
Transformation is rigid since the structure of the eye vessels will not change from one frame to another.

Each video file read belongs to one patient.
### Used Methods 
- Transformation Type: Rigid
- Interpolation: Cubic
- Similarity Function: Normalized Cross-Corrolation
- Optimizer: Will be determined..



