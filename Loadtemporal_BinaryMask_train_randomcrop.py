from __future__ import print_function, division
import os
import torch
import pandas as pd
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
import imgaug.augmenters as iaa
from utils import get_face_box

import cv2_trans
import h5py




# data augment from 'imgaug' --> Add (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
seq = iaa.Sequential([
    iaa.Add(value=(-40,40), per_channel=True), # Add color 
    iaa.GammaContrast(gamma=(0.5,1.5)) # GammaContrast with a gamma of 0.5 to 1.5
])





# array
class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability = 0.5, sl = 0.01, sh = 0.05, r1 = 0.5, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, sample):
        img, binary_mask, spoofing_label = sample['image_x'], sample['binary_mask'],sample['spoofing_label']
        
        if random.uniform(0, 1) < self.probability:
            attempts = np.random.randint(1, 3)
            for attempt in range(attempts):
                area = img.shape[0] * img.shape[1]
           
                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1/self.r1)
    
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
    
                if w < img.shape[1] and h < img.shape[0]:
                    x1 = random.randint(0, img.shape[0] - h)
                    y1 = random.randint(0, img.shape[1] - w)

                    img[x1:x1+h, y1:y1+w, 0] = self.mean[0]
                    img[x1:x1+h, y1:y1+w, 1] = self.mean[1]
                    img[x1:x1+h, y1:y1+w, 2] = self.mean[2]
                    
        return {'image_x': img, 'binary_mask': binary_mask, 'spoofing_label': spoofing_label}


# Tensor
class Cutout(object):
    def __init__(self, length=50):
        self.length = length

    def __call__(self, sample):
        img, binary_mask, spoofing_label = sample['image_x'], sample['binary_mask'],sample['spoofing_label']
        h, w = img.shape[1], img.shape[2]    # Tensor [1][2],  nparray [0][1]
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        length_new = np.random.randint(1, self.length)
        
        y1 = np.clip(y - length_new // 2, 0, h)
        y2 = np.clip(y + length_new // 2, 0, h)
        x1 = np.clip(x - length_new // 2, 0, w)
        x2 = np.clip(x + length_new // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return {'image_x': img, 'binary_mask': binary_mask, 'spoofing_label': spoofing_label}


class Normaliztion(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    def __call__(self, sample):
        image_x, binary_mask, spoofing_label = sample['image_x'], sample['binary_mask'],sample['spoofing_label']
        
        new_image_x = (image_x - 127.5)/128     # [-1,1]

        return {'image_x': new_image_x, 'binary_mask': binary_mask, 'spoofing_label': spoofing_label}



class RandomHorizontalFlip(object):
    """Horizontally flip the given Image randomly with a probability of 0.5."""
    def __call__(self, sample):
        image_x, binary_mask, spoofing_label = sample['image_x'], sample['binary_mask'],sample['spoofing_label']
        
        new_image_x = np.zeros_like(image_x)
        new_binary_mask = np.zeros_like(binary_mask)

        p = random.random()
        if p < 0.5:
            #print('Flip')
            new_image_x = cv2.flip(image_x, 1)
            new_binary_mask = cv2.flip(binary_mask, 1)
           
                
            return {'image_x': new_image_x, 'binary_mask': new_binary_mask, 'spoofing_label': spoofing_label}
        else:
            #print('no Flip')
            return {'image_x': image_x, 'binary_mask': binary_mask, 'spoofing_label': spoofing_label}


    
class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        image_x, binary_mask, spoofing_label = sample['image_x'], sample['binary_mask'],sample['spoofing_label']
        
        # swap color axis because
        # numpy image: (batch_size) x H x W x C
        # torch image: (batch_size) x C X H X W
        image_x = image_x[:,:,::-1].transpose((2, 0, 1))
        image_x = np.array(image_x)
        
        binary_mask = np.array(binary_mask)

                        
        spoofing_label_np = np.array([0],dtype=np.compat.long)
        spoofing_label_np[0] = spoofing_label
        
        
        return {'image_x': torch.from_numpy(image_x.astype(float)).float(), 'binary_mask': torch.from_numpy(binary_mask.astype(float)).float(), 'spoofing_label': torch.from_numpy(spoofing_label_np.astype(float)).float()}



def _read_img_lmdb(env, key):
    """read image from lmdb with key (w/ and w/o fixed size)
    """
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    buf = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


class Spoofing_train_LmdbDS(Dataset):
    def __init__(self, info_list, lmdb_dir, ldmk_dir='train_landmark_R50', illupath=None, colorpath=None, transform=None, debug=None, crop_size=64, map_size=(16,16), binary_type = 'gray'):

        with open(info_list, 'r') as f:
            self.landmarks_frame = f.readlines()
        if illupath != None:
            self.illulst = h5py.File(illupath, 'r')['data']
        if colorpath != None:
            self.colorlst = h5py.File(colorpath, 'r')['data']
        self.root = lmdb_dir
        self.len = len(self.landmarks_frame)
        self.map_size = map_size
        self.crop_size = crop_size
        self.landmarks_frame[:1000]
        self.transform = transform
        self.binary_type = binary_type
        self.ldmk_dir = ldmk_dir

    def __len__(self):
        return len(self.landmarks_frame)
    
    def __getitem__(self, idx):
        spts = self.landmarks_frame[idx].strip().split(' ')
        spoofing_label = int(spts[1])
        file_name = spts[0]
        lmk_name = '/'.join(file_name.split('/')[-2:]).split('.')[0]
        lmk_dir = os.path.join(self.ldmk_dir, lmk_name + '.txt')
        image_x = cv2.imread(os.path.join(self.root, file_name))
        
        h, w, _ = image_x.shape
        image_x = cv2_trans.random_resize(image_x, 0.7, 0.7, 1)
        image_x = cv2_trans.random_jpeg(image_x, 0.7, 70, 100)
        if self.illulst != None:
            image_x = cv2_trans.random_illu(image_x, 0.1, self.illulst)
        if self.colorlst != None:
            image_x = cv2_trans.random_color(image_x, 0.1, self.colorlst)
        image_x = cv2.resize(image_x, (h, w))
        
        assert image_x is not None, "{} not exist!".format(os.path.join(self.root, file_name))
        h, w, _ = image_x.shape
        binary_mask_canvas = cv2.cvtColor(image_x, cv2.COLOR_BGR2GRAY)
        if os.path.exists(os.path.join(self.root, lmk_dir)):
            square_box = get_face_box(os.path.join(self.root, lmk_dir), h, w, True)
            image_x = image_x[square_box[1]:square_box[3],square_box[0]:square_box[2]]
            ori_box = get_face_box(os.path.join(self.root, lmk_dir), h, w, False)
        else:
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
        binary_mask = cv2.resize(binary_mask.astype(np.uint8), self.map_size, interpolation=cv2.INTER_NEAREST)  if spoofing_label == 1 else np.zeros(self.map_size)
            
        image_x = cv2.resize(image_x, (self.crop_size, self.crop_size))
        image_x_aug = seq.augment_image(image_x)
        
        
        sample = {'image_x': image_x_aug, 'binary_mask': binary_mask, 'spoofing_label': spoofing_label}

        if self.transform:
            sample = self.transform(sample)
        return sample




if __name__ == '__main__':
    from tqdm import tqdm
    train_list = '/ssd/zhengmingwu/iccv_dataset/train_label_balance.txt' 
    root_dir = '/ssd/zhengmingwu/iccv_dataset/'
    illupath = '/ssd/zhengmingwu/iccv/illu.h5'
    colorpath = '/ssd/zhengmingwu/iccv/color.h5'
    val_data = Spoofing_train_LmdbDS(train_list, root_dir, illupath=illupath, colorpath=colorpath,  transform=transforms.Compose([RandomErasing(), RandomHorizontalFlip(),  ToTensor(), Cutout(), Normaliztion()]), crop_size=256, map_size=(32,32))
    dataloader_val = DataLoader(val_data, batch_size=1, shuffle=True, num_workers=8)

    for i, sample_batched in enumerate(tqdm(dataloader_val)):
        pass
    quit()