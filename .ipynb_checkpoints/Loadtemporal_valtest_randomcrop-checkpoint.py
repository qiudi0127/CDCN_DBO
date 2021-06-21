from __future__ import print_function, division
import pandas as pd
import os
import torch

#from skimage import io, transform
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pdb
import math
import os 
import lmdb
from utils import get_face_box
from glob import glob

# frames_total = 8    # each video 8 uniform samples
total = 0
spoof = 0

class Normaliztion_valtest(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    def __call__(self, sample):
        image_x, binary_mask, spoofing_label = sample['image_x'],sample['binary_mask'],sample['spoofing_label']
        new_image_x = (image_x - 127.5)/128     # [-1,1]
        
        return {'image_x': new_image_x, 'binary_mask': binary_mask, 'spoofing_label': spoofing_label, 'ori_image': sample['ori_image'], 'file_info': sample['file_info'], 'noface': sample['noface']}


class ToTensor_valtest(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        image_x, binary_mask, spoofing_label = sample['image_x'],sample['binary_mask'],sample['spoofing_label']
        
        # swap color axis because    BGR2RGB
        # numpy image: (batch_size) x T x H x W x C
        # torch image: (batch_size) x T x C X H X W
        # image_x = image_x[:,:,:,::-1].transpose((0, 3, 1, 2))
        # image_x = np.array(image_x)

        image_x = image_x[:,:,::-1].transpose((2, 0, 1))
        image_x = np.array(image_x)
                        
        binary_mask = np.array(binary_mask)

        #spoofing_label_np = np.array([0],dtype=np.long)
        #spoofing_label_np[0] = spoofing_label
        
        return {'image_x': torch.from_numpy(image_x.astype(np.float)).float(), 'binary_mask': torch.from_numpy(binary_mask.astype(np.float)).float(), 'spoofing_label':spoofing_label,'ori_image': sample['ori_image'], 'file_info': sample['file_info'], 'noface': sample['noface']} 

def _read_img_lmdb(env, key):
    """read image from lmdb with key (w/ and w/o fixed size)
    """
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    buf = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


class Spoofing_valtest_LmdbDS(Dataset):

    def __init__(self, info_list, lmdb_dir, ldmk_dir='val_landmark_R50', transform=None, debug = None, crop_size=128, map_size=(16,16), binary_type = 'gray'):
        with open(info_list, 'r') as f:
            self.landmarks_frame = f.readlines()

        self.root = lmdb_dir
        self.len = len(self.landmarks_frame)
        self.crop_size = crop_size
        self.map_size = map_size
        self.binary_type = binary_type
        self.transform = transform
        self.ldmk_dir = ldmk_dir

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        noface = False
        spts = self.landmarks_frame[idx].strip().split(' ')
        file_name = spts[0]
        spoofing_label = int(spts[1])
        lmk_name = '/'.join(file_name.split('/')[-2:]).split('.')[0]
        lmk_dir = os.path.join(self.ldmk_dir, lmk_name + '.txt')
        image_x = cv2.imread(os.path.join(self.root,file_name))
        assert image_x is not None,'{} not exist!'.format(os.path.join(self.root,file_name))
        h, w, _ = image_x.shape
        binary_mask_canvas = cv2.cvtColor(image_x, cv2.COLOR_BGR2GRAY)
        
        if os.path.exists(os.path.join(self.root, lmk_dir)):
            square_box = get_face_box(os.path.join(self.root, lmk_dir), h, w, True)
            image_x = image_x[square_box[1]:square_box[3],square_box[0]:square_box[2]]
            ori_box = get_face_box(os.path.join(self.root, lmk_dir), h, w, False)
        else:
            noface = True
            image_x = image_x
            square_box = [0, 0, h, w]
            ori_box = [0, 0, h, w]

            
        if self.binary_type == 'gray':
            binary_mask_canvas = np.where(binary_mask_canvas > 0, 1, 0)
        elif self.binary_type == 'black':
            binary_mask_canvas = np.ones(self.map_size)
        else: # 'center'
            binary_mask_canvas = np.zeros_like(binary_mask_canvas)
            binary_mask_canvas[ori_box[1]:ori_box[3],ori_box[0]:ori_box[2]] = 1
        binary_mask = binary_mask_canvas[square_box[1]:square_box[3],square_box[0]:square_box[2]]
        binary_mask = cv2.resize(binary_mask.astype(np.uint8), self.map_size, interpolation=cv2.INTER_NEAREST) 
            
        image_x = cv2.resize(image_x, (self.crop_size, self.crop_size))
        sample = {'image_x': image_x, 'binary_mask': binary_mask, 'spoofing_label': spoofing_label, 'ori_image': cv2.cvtColor(image_x,cv2.COLOR_BGR2RGB).transpose((2,0,1)), 'file_info': file_name, 'noface': noface}

        if self.transform:
            sample = self.transform(sample)
            
        return sample

    
    
class Spoofing_valtest_fin(Dataset):

    def __init__(self, test_dir, landmark_dir, transform=None, crop_size=128, map_size=(16,16), binary_type = 'gray'):
        self.tests = glob(os.path.join(test_dir, '*', '*.png'))
        self.landmark_dir = landmark_dir
        self.crop_size = crop_size
        self.map_size = map_size
        self.binary_type = binary_type
        self.transform = transform

    def __len__(self):
        return len(self.tests)

    def __getitem__(self, idx):
        noface = False
        png = self.tests[idx]
        dir_name = png.split('/')[-2]
        png_name = png.split('/')[-1]
        
        lmk_name = os.path.join(self.landmark_dir ,dir_name, png_name.replace('png','txt'))
        image_x = cv2.imread(png)
        assert image_x is not None,'{} not exist!'.format(os.path.join(self.root,file_name))
        h, w, _ = image_x.shape
         
        binary_mask_canvas = cv2.cvtColor(image_x, cv2.COLOR_BGR2GRAY)
        if os.path.exists(lmk_name):
            square_box = get_face_box(lmk_name, h, w, True)
            image_x = image_x[square_box[1]:square_box[3],square_box[0]:square_box[2]]
            ori_box = get_face_box(lmk_name, h, w, False)
        else:
            noface = True
            image_x = image_x
            square_box = [0, 0, h, w]
            ori_box = [0, 0, h, w]

        if self.binary_type == 'gray':
            binary_mask_canvas = np.where(binary_mask_canvas > 0, 1, 0)
        elif self.binary_type == 'black':
            binary_mask_canvas = np.ones(self.map_size)
        else: # 'center'
            binary_mask_canvas = np.zeros_like(binary_mask_canvas)
            binary_mask_canvas[ori_box[1]:ori_box[3],ori_box[0]:ori_box[2]] = 1
        binary_mask = binary_mask_canvas[square_box[1]:square_box[3],square_box[0]:square_box[2]]
        binary_mask = cv2.resize(binary_mask.astype(np.uint8), self.map_size, interpolation=cv2.INTER_NEAREST) 
            
        image_x = cv2.resize(image_x, (self.crop_size, self.crop_size))
        
        sample = {'image_x': image_x, 'binary_mask': binary_mask, 'spoofing_label': 1, 'ori_image': cv2.cvtColor(image_x,cv2.COLOR_BGR2RGB).transpose((2,0,1)), 'file_info': png_name, 'noface': noface}

        if self.transform:
            sample = self.transform(sample)
            
        return sample
    

total = 0
spoof = 0
if __name__ == '__main__':
    
    pass
