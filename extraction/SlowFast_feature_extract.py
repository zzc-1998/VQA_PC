import os
import argparse
import cv2
import pandas as pd
import numpy as np

import torch
from torch.utils import data
from torchvision import transforms
from pytorchvideo.models.hub import slowfast_r50
import torch.nn as nn

from PIL import Image



class Dataset_slowfast_feature(data.Dataset):
    """Read data from the frames for feature extraction"""
    def __init__(self, data_dir, transform, resize):
        super(Dataset_slowfast_feature, self).__init__()
        self.frames_dir = data_dir
        self.video_names = os.listdir(data_dir)
        self.transform = transform
        self.resize = resize
        self.length = len(self.video_names)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        
        # 4 clips x 30 frames =120 frames
        video_channel = 3
        video_frame_rate = 30
        video_read_interval = video_frame_rate
        video_clip_read = 4

        # fit the slowfast 32 frames input requirement
        video_length_clip = 32
        video_length = video_clip_read*video_read_interval

        transformed_frame_all = torch.zeros([video_length, video_channel, self.resize, self.resize])
        transformed_video_all = []
        
        for i in range(video_length):
            img_name = str(i+1).zfill(3) + '.png'
            filename=os.path.join(self.frames_dir, video_name, img_name)
            read_frame = Image.fromarray(cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB))
            read_frame = self.transform(read_frame)
            transformed_frame_all[i] = read_frame

        video_clip_read_index = 0
        for i in range(video_clip_read):
            transformed_video = torch.zeros([video_length_clip, video_channel, self.resize, self.resize])
            if (i*video_read_interval + video_length_clip) <= video_length:
                transformed_video = transformed_frame_all[i*video_read_interval : (i*video_read_interval + video_length_clip)]
            else:
                transformed_video[:(video_length - i*video_read_interval)] = transformed_frame_all[i*video_read_interval:]
                for j in range((video_length - i*video_read_interval), video_length_clip):
                    transformed_video[j] = transformed_video[video_length - i*video_read_interval - 1]
            video_clip_read_index += 1
            transformed_video_all.append(transformed_video)
        # video_clip_read * video_frame_rate * color_channel * width * height 
        return transformed_video_all, video_name

def pack_pathway_output(frames, device):
    """
    Prepare output as a list of tensors. Each tensor corresponding to a
    unique pathway.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
    Returns:
        frame_list (list): list of tensors with the dimension of
            `channel` x `num frames` x `height` x `width`.
    """

    fast_pathway = frames
    # Perform temporal sampling from the fast pathway.
    
    slow_pathway = torch.index_select(
        frames,
        2,
        torch.linspace(
            0, frames.shape[2] - 1, frames.shape[2] // 4
        ).long(),
    )
    frame_list = [slow_pathway.to(device), fast_pathway.to(device)]

    return frame_list


class slowfast(torch.nn.Module):
    def __init__(self):
        super(slowfast, self).__init__()
        slowfast_pretrained_features = nn.Sequential(*list(slowfast_r50(pretrained=True).children())[0])

        self.feature_extraction = torch.nn.Sequential()
        self.slow_avg_pool = torch.nn.Sequential()
        self.fast_avg_pool = torch.nn.Sequential()
        self.adp_avg_pool = torch.nn.Sequential()

        for x in range(0,5):
            self.feature_extraction.add_module(str(x), slowfast_pretrained_features[x])

        self.slow_avg_pool.add_module('slow_avg_pool', slowfast_pretrained_features[5].pool[0])
        self.fast_avg_pool.add_module('fast_avg_pool', slowfast_pretrained_features[5].pool[1])
        self.adp_avg_pool.add_module('adp_avg_pool', slowfast_pretrained_features[6].output_pool)
        

    def forward(self, x):
        
        with torch.no_grad():
            x = self.feature_extraction(x)

            slow_feature = self.slow_avg_pool(x[0])
            fast_feature = self.fast_avg_pool(x[1])

            slow_feature = self.adp_avg_pool(slow_feature)
            fast_feature = self.adp_avg_pool(fast_feature)
            
        return slow_feature, fast_feature

def main(config):

    if not os.path.exists(config.feature_save_folder):
                os.makedirs(config.feature_save_folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = slowfast()
    model = model.to(device)

    
        
    ## loading data
    resize = config.resize
    frames_dir = config.frames_dir
    transformations_test = transforms.Compose([transforms.CenterCrop([resize, resize]),transforms.ToTensor(),\
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    trainset = Dataset_slowfast_feature(frames_dir, transformations_test, resize)
    ## dataloader
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=1,
        shuffle=False, num_workers=config.num_workers)

    # extracting features
    with torch.no_grad():
        model.eval()
        for i, (video, video_name) in enumerate(train_loader):
            video_name = video_name[0].split('.')[0]
            print(video_name)
            if not os.path.exists(config.feature_save_folder + video_name):
                os.makedirs(config.feature_save_folder + video_name)
            
            for idx, ele in enumerate(video):
                ele = ele.permute(0, 2, 1, 3, 4)
                ele = pack_pathway_output(ele, device)
                _, fast_feature = model(ele)
                np.save(config.feature_save_folder + video_name + '/' + 'feature_' + str(idx) + '_fast_feature', fast_feature.to('cpu').numpy())

        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resize', type=int, default=224)
    parser.add_argument('--frames_dir', type=str, default='') # if you use the rotation.py, here should be path to the 'path to ./rotation/frames/'
    parser.add_argument('--feature_save_folder', type=str, default='./features/') #path to save fast features

    config = parser.parse_args()

    main(config)
