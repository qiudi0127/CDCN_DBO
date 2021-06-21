from __future__ import print_function, division
#import pandas as pd
from utils import AvgrageMeter, accuracy, performances_Cele

import torch
import argparse,os
import cv2
import numpy as np
import random
import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

#from models.CDCNs import Conv2d_cd, CDCNpp
from Loadtemporal_valtest_randomcrop import Spoofing_valtest_fin, Normaliztion_valtest, ToTensor_valtest, Spoofing_valtest_LmdbDS

import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import copy
import pdb
#from models.CDCNs_192_more import Conv2d_cd, CDCNpp
from models.BilateralConv_more import Conv2d_cd, BCNpp
#from utils import AvgrageMeter, accuracy, performances_Cele
# from models.BilateralConv import Conv2d_cd, BCNpp

from tqdm import tqdm

# Dataset root     
val_image_dir = 'dataset'  
val_test_list =  'dataset/val_label.txt'

# main function
def train_test(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if not os.path.exists(args.log):
        os.makedirs(args.log)
    #model = cbam_resnet50() 
    #model = CDCNpp( basic_conv=Conv2d_cd, theta=0.5, map_size=(16,16))
    #model = CDCNpp( basic_conv=Conv2d_cd, theta=args.theta)
    model = BCNpp(basic_conv=Conv2d_cd, theta=0.5, map_size=(16,16))

    model = nn.DataParallel(model)
    # print(args.resume)
    save_param = torch.load(args.resume)
    model.load_state_dict(save_param['state_dict'])
    
    model = model.cuda()

    #print(model) 
    
    model.eval()
    
    with torch.no_grad():
        
        ###########################################
        '''                val             '''
        ###########################################
        val_data = Spoofing_valtest_LmdbDS(val_test_list, val_image_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]), crop_size=128, map_size=(16,16))
        dataloader_val = DataLoader(val_data, batch_size=16, shuffle=False, num_workers=8)
        
        map_score_list_val = []
        
        for i, sample_batched in enumerate(tqdm(dataloader_val)):
            inputs = sample_batched['image_x'].cuda()
            spoofing_label,binary_mask = sample_batched['spoofing_label'], sample_batched['binary_mask'].cuda()

            map_score = 0.0
            map_x =  model(inputs)
            for i in range(map_x.shape[0]):
                t_map = map_x[i]
                map_score = torch.sum(t_map)/torch.sum(binary_mask[i])
                if map_score>1:
                    map_score = 1.0
                # if sample_batched['noface'][i] == True:
                #     map_score = 0.0
                map_score_list_val.append('{} {}\n'.format(os.path.basename(sample_batched['file_info'][i]), map_score))
            
        map_score_list_val.sort(key=sort_key)
        # print(map_score_list_val)
        ###########################################
        '''                test             '''
        ###########################################
        # val for threshold
        test_data = Spoofing_valtest_fin(args.image_dir, args.ldmk_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]), crop_size=128, map_size=(16,16))
        dataloader_test = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=8)
        
        map_score_list_test = []
        
        for i, sample_batched in enumerate(tqdm(dataloader_test)):
            
            inputs = sample_batched['image_x'].cuda()
            png_names = sample_batched['file_info']
            spoofing_label,binary_mask = sample_batched['spoofing_label'], sample_batched['binary_mask'].cuda()

            map_score = 0.0
            map_x =  model(inputs)
            for i in range(map_x.shape[0]):
                t_map = map_x[i]
                map_score = torch.sum(t_map)/torch.sum(binary_mask[i])
                if map_score>1:
                    map_score = 1.0
                # if sample_batched['noface'][i] == True:
                #     map_score = 0.0
                map_score_list_test.append('{} {}\n'.format(png_names[i], map_score))
            
        map_score_list_test.sort(key=sort_key)
        
        map_score_val_filename = os.path.join(args.log ,'map_score_test_{}.txt'.format(args.epoch))
        
        with open(map_score_val_filename, 'w') as file:
            file.writelines(map_score_list_val)
            file.writelines(map_score_list_test)

def sort_key(ele):
    return int(ele.split('.')[0])
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--gpu', type=str, default='0,1', help='the gpu id used for predict')
    parser.add_argument('--epoch', type=int, default=50, help='total training epochs')
    parser.add_argument('--resume', type=str, default='', help='')
    parser.add_argument('--log', type=str, default="BCN", help='log and save model name')
    parser.add_argument('--test_num', type=int, default=1000, help='total training epochs')
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--ldmk_dir', type=str)
    args = parser.parse_args()
    train_test(args)
