# VQA_PC
Treating point cloud as moving camera videos: a no-reference quality assessment metric 

<img align="center" src="https://github.com/zzc-1998/VQA_PC/blob/main/video.gif">

## Environment Settings
We test the code with Python 3.7 (and higher) on the Windows platform and the code can run on linux as well.

You should install the python package open3d and pytorch >= 1.6.

## Start with Generating 2D Inputs and Frames
Use the **rotation.py** in folder **rotation**. We have prepared two .ply samples for test. You can simply run the **rotation.py** with default parameters for test. The **rotation.py** should generate the 2D Inputs, frames, and videos in the *./imgs*, *./frames*, and *./videos* located in the same dir as **rotation.py**.

To test with your data, please change the necessray input and output as illustrated:

```
parser.add_argument('--path', type=str, default = './ply/') #path to the file that contain .ply models
parser.add_argument('--img_path', type=str, default = './imgs/') # path to the generated 2D input
parser.add_argument('--frame_path', type=str, default = './frames/') # path to the generated frames
parser.add_argument('--video_path', type=str, default = './videos/') # path to the generated videos
```
By running this demo, you wil get three output folders. **imgs** contain 4 2D inputs with resolution of 1920x1061. **frames** contain 120 frames with resized resolution of 405x224. **videos** contain the videos generated from the frames.

## Extracting SlowFast Features
To avoid compression loss, we suggest extracting 3D features directly from the rendered frames.
