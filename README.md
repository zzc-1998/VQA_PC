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
We suggest extracting 3D features directly from the rendered frames. Use the **SlowFast_feature_extract.py** in the extraction folder to extarct *fast* features. If the **rotation.py** is used, the *frames_dir* should be the *path to ./rotation/frames/*.
```
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--resize', type=int, default=224)
parser.add_argument('--frames_dir', type=str, default='path to frames') # if you use the rotation.py, here should be path to the 'path to ./rotation/frames/'
parser.add_argument('--feature_save_folder', type=str, default='./features/') #path to save fast features
```


## Test
Run the **test.py** in the test folder. We provide the trained checkpoint, extarcted 2d inputs, and slowfast features [google drive here](https://drive.google.com/drive/folders/1-z-X0K3qOPF3swr79kKqmKZjXafwxJu3?usp=sharing). Use the command to test on the SJTU-PCQA database as follows:
```
python -u test.py  \
--pretrained_model_path ''path/to/ResNet_mean_fast_SJTU.pth'' \
--path_imgs ''path/to/sjtu_2d/'' \
--path_3d_features ''path/to/sjtu_slowfast''  \
--data_info  data_info/sjtu_mos.csv \
--output_csv_path sjtu_prediction.csv 
```
The predicted results are saved in .csv file. Change the necessary parameters if you want to test on the WPC database.


## Train
To begin training, please download the images and slowfast features from [google drive here](https://drive.google.com/drive/folders/1-z-X0K3qOPF3swr79kKqmKZjXafwxJu3?usp=sharing) and unzip the files to .train/database/:
```
train----database------sjtu_2d---hhi_0.ply--|-005.png
                   |                        |-035.png
                   |                        |-065.png
                   |                        |-095.png
                   |---sjtu_datainfo
                   |---sjtu_slowfast---hhi_0--|-feature_0_fast_feature.npy
                                              |-feature_1_fast_feature.npy
                                              |-feature_2_fast_feature.npy
                                              |-feature_3_fast_feature.npy
                   |---wpc_2d
                   |---wpc_datainfo
                   |---wpc_slowfast
                                                                      
```
Run the training model on the SJTU-PCQA and WPC databases by using the following command:
```
CUDA_VISIBLE_DEVICES=0 python -u train.py \
 --database SJTU \
 --model_name  ResNet_mean_with_fast \
 --split_num 9 \
 --conv_base_lr 0.00004 \
 --decay_ratio 0.9 \
 --decay_interval 10 \
 --train_batch_size 32 \
 --num_workers 6 \
 --epochs 30 \
 --split_num 9 \
 --ckpt_path ckpts \
 >> logs/sjtu.log  
```
```
CUDA_VISIBLE_DEVICES=0 python -u train.py \
 --database WPC \
 --model_name  ResNet_mean_with_fast \
 --conv_base_lr 0.00005 \
 --decay_ratio 0.9 \
 --decay_interval 10 \
 --train_batch_size 32 \
 --num_workers 6 \
 --epochs 30 \
 --split_num 5 \
 >> logs/wpc.log  
```

# Citation
If you find our work useful, please cite our paper as:
```
```



