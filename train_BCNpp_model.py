from __future__ import print_function, division
from utils import AvgrageMeter, accuracy, performances_Cele 
import pandas as pd
import torch
#import matplotlib as mpl
#mpl.use('TkAgg')
import argparse,os
#import pandas as pd
import cv2
import numpy as np
import random
import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from models.BilateralConv import Conv2d_cd, BCNpp

from Loadtemporal_BinaryMask_train_randomcrop import Spoofing_train_LmdbDS, Normaliztion, ToTensor, RandomHorizontalFlip, Cutout, RandomErasing
from Loadtemporal_valtest_randomcrop import Spoofing_valtest_LmdbDS, Normaliztion_valtest, ToTensor_valtest


import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import copy
import pdb
from tqdm import tqdm
#from utils import AvgrageMeter, accuracy, performances

def contrast_depth_conv(input):
    ''' compute contrast depth in both of (out, label) '''
    '''
        input  32x32
        output 8x32x32
    '''
    kernel_filter_list =[
                            [[1,0,0],[0,-1,0],[0,0,0]], [[0,1,0],[0,-1,0],[0,0,0]], [[0,0,1],[0,-1,0],[0,0,0]],
                            [[0,0,0],[1,-1,0],[0,0,0]], [[0,0,0],[0,-1,1],[0,0,0]], [[0,0,0],[0,-1,0],[1,0,0]], 
                            [[0,0,0],[0,-1,0],[0,1,0]], [[0,0,0],[0,-1,0],[0,0,1]]
                        ]
    kernel_filter = np.array(kernel_filter_list, np.float32)
    kernel_filter = torch.from_numpy(kernel_filter.astype(float)).float().to(torch.device('cuda'))
    kernel_filter = kernel_filter.unsqueeze(dim=1)
    input = input.unsqueeze(dim=1).expand(input.shape[0], 8, input.shape[1],input.shape[2])
    contrast_depth = F.conv2d(input, weight=kernel_filter, groups=8)  # depthwise conv
    
    return contrast_depth


class Contrast_depth_loss(nn.Module):    # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Contrast_depth_loss,self).__init__()
        return
    def forward(self, out, label): 
        '''
        compute contrast depth in both of (out, label),
        then get the loss of them
        tf.atrous_convd match tf-versions: 1.4
        '''
        contrast_out = contrast_depth_conv(out)
        contrast_label = contrast_depth_conv(label)
        criterion_MSE = nn.MSELoss().to(torch.device('cuda'))
        loss = criterion_MSE(contrast_out, contrast_label)
 
        return loss

def adjust_learning_rate(args,optimizer,lr):
    """Sets the learning rate to the initial LR decayed by 10 every step epochs"""
    lr *= args.gamma
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr
    return lr



# main function
def train_test():
    def logging(str):
        print(str)
        log_file.write(str)
        log_file.flush()
    # GPU  & log file  -->   if use DataParallel, please comment this command
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda')
    
    isExists = os.path.exists(args.log)
    if not isExists:
        os.makedirs(args.log)
    log_file = open(args.log+'/'+ args.log+'_log.txt', 'a+')
    
    echo_batches = args.echo_batches

    finetune = args.finetune
    if finetune==True:
        logging('Finetune from epoch: {}!\n'.format(args.restore_epoch))
        model = BCNpp(basic_conv=eval(args.basic_conv), theta=args.theta, map_size=(args.map_size,args.map_size))
        model.to(torch.device('cuda'))
        model = nn.DataParallel(model)
        lr = args.lr
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00005)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        save_param = torch.load('{}/{}_{}.pth'.format(args.log, args.log, args.restore_epoch))
        model.load_state_dict(save_param['state_dict'])
        optimizer.load_state_dict(save_param['optimizer'])
        for i in range(args.restore_epoch):
            if (i + 1) % args.step_size == 0:
                lr = adjust_learning_rate(args,optimizer,lr)
            scheduler.step()
    else:
        print('train from scratch!\n')
        log_file.write('train from scratch!\n')
        log_file.flush()
         
        model = BCNpp(basic_conv=eval(args.basic_conv), theta=args.theta, map_size=(args.map_size, args.map_size))
        model = model.to(torch.device('cuda'))
        model = nn.DataParallel(model)

        lr = args.lr
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00005)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    # print(model) 
    criterion_absolute_loss = nn.MSELoss()
    criterion_contrastive_loss = Contrast_depth_loss()
    acer_save = 1.0
    
    for epoch in range(args.restore_epoch,args.epochs):  # loop over the dataset multiple times
        loss_absolute = AvgrageMeter()
        loss_contra =  AvgrageMeter()
        ###########################################
        '''                train             '''
        ###########################################
        model.train()
        
        
        train_data = Spoofing_train_LmdbDS(args.train_list, args.root, illupath=args.illupath, colorpath=args.colorpath, transform=transforms.Compose([RandomErasing(), RandomHorizontalFlip(),  ToTensor(), Cutout(), Normaliztion()]), crop_size=args.crop_size, map_size=(args.map_size,args.map_size), binary_type=args.binary_type)
        dataloader_train = DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=10)

        for i, sample_batched in enumerate(tqdm(dataloader_train)):
            # get the inputs
            inputs, binary_mask, spoof_label = sample_batched['image_x'].to(device), sample_batched['binary_mask'].to(device), sample_batched['spoofing_label'].to(device)

            optimizer.zero_grad()

            map_x =  model(inputs)
            absolute_loss = criterion_absolute_loss(map_x, binary_mask)
            contrastive_loss = criterion_contrastive_loss(map_x, binary_mask)
            loss =  absolute_loss + contrastive_loss
             
            loss.backward()
            optimizer.step()
            
            n = inputs.size(0)
            loss_absolute.update(absolute_loss.data, n)
            loss_contra.update(contrastive_loss.data, n)
       
        scheduler.step()
        if (epoch + 1) % args.step_size == 0:
            lr *= args.gamma
        print('Epoch: %d, Train: Absolute_Depth_loss = %.4f, Contrastive_Depth_loss = %.4f' % (epoch + 1, loss_absolute.avg, loss_contra.avg))
        log_file.write('Epoch: %d, Train: Absolute_Depth_loss = %.4f, Contrastive_Depth_loss = %.4f \n' % (epoch + 1, loss_absolute.avg, loss_contra.avg))
        log_file.flush()
           
            
        epoch_test = 1  # epoch test interval
        if epoch % epoch_test == epoch_test-1:    
            model.eval()
            with torch.no_grad():
                ###########################################
                '''                val             '''
                ###########################################
                val_data = Spoofing_valtest_LmdbDS(args.val_list, args.root, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]),debug=args.debug, crop_size=args.crop_size, map_size=(args.map_size,args.map_size), binary_type=args.binary_type)
                dataloader_val = DataLoader(val_data, batch_size=args.batchsize, shuffle=False, num_workers=8)
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_val):
                    # get the inputs
                    inputs = sample_batched['image_x'].to(device)
                    spoofing_label, binary_mask = sample_batched['spoofing_label'], sample_batched['binary_mask'].to(device)
        
                    optimizer.zero_grad()
                    map_x =  model(inputs)
                    for i in range(map_x.shape[0]):
                        t_map = map_x[i]
                        map_score = torch.sum(t_map)/torch.sum(binary_mask[i])
                        if map_score>1:
                            map_score = 1.0
                        map_score_list.append('{0:.4f} {1}\n'.format( map_score,spoofing_label.numpy()[i]))

                map_score_val_filename = args.log+'/'+ args.log+ '_map_score_val_randomcrop64_c160_%d.txt'% (epoch + 1)
                with open(map_score_val_filename, 'w') as file:
                    file.writelines(map_score_list)                
                
                val_threshold, val_ACC, val_ACER, val_APCER,val_BPCER = performances_Cele(map_score_val_filename)
                if val_ACER < acer_save:
                    acer_save = val_ACER
                log_file.write('val_threshold:{0:.4f} val_ACC:{1:.4f} val_ACER:{2:.4f} val_APCER:{3:.4f} val_BPCER:{4:.4f} best_acer:{5:.4f}\n'.format(val_threshold, val_ACC, val_ACER,val_APCER,val_BPCER,acer_save))
                log_file.flush()
                print('val_threshold:{0:.4f} val_ACC:{1:.4f} val_ACER:{2:.4f} val_APCER:{3:.4f} val_BPCER:{4:.4f} best_acer:{5:.4f}'.format(val_threshold, val_ACC, val_ACER,val_APCER,val_BPCER,acer_save))
                
            #torch.save(model.state_dict(), args.log+'/'+args.log+'_%d.pth' % (epoch + 1))
        save_param = {
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict()
        }
        torch.save(save_param, os.path.join(args.log, args.log + '_{:0>3d}.pth'.format(epoch + 1)))

    print('Finished Training')
    log_file.close()
  

  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--gpu', type=str, default='3', help='the gpu id used for predict')
    parser.add_argument('--lr', type=float, default=0.00008, help='initial learning rate')  #default=0.0001
    parser.add_argument('--batchsize', type=int, default=9, help='initial batchsize')  #default=7  
    parser.add_argument('--step_size', type=int, default=20, help='how many epochs lr decays once')  # 500  | DPC = 400
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--echo_batches', type=int, default=50, help='how many batches display once')  # 50
    parser.add_argument('--epochs', type=int, default=60, help='total training epochs')
    parser.add_argument('--restore_epoch', type=int, default=0, help='restore epoch')
    parser.add_argument('--log', type=str, default="CDCNpp_BinaryMask_Cele", help='log and save model name')
    parser.add_argument('--finetune', action='store_true', default=False, help='whether finetune other models')
    parser.add_argument('--theta', type=float, default=0.7, help='hyper-parameters in CDCNpp')
    parser.add_argument('--train_num', type=int, default=-1, help='hyper-parameters in CDCNpp')
    parser.add_argument('--debug', type=str, default=None, help='hyper-parameters in CDCNpp')
    parser.add_argument('--map_size', type=int, default=None, help='')
    parser.add_argument('--crop_size', type=int, default=64, help='')
    parser.add_argument('--train_list', type=str, default='', help='')
    parser.add_argument('--val_list', type=str, default='', help='')
    parser.add_argument('--root', type=str, default='', help='')
    parser.add_argument('--basic_conv', type=str, default='Conv2d_cd', help='')
    parser.add_argument('--binary_type', type=str, default='gray', help='')
    parser.add_argument('--illupath', type=str, default='/ssd/zhengmingwu/iccv/illu.h5', help='')
    parser.add_argument('--colorpath', type=str, default='/ssd/zhengmingwu/iccv/color.h5', help='')
    parser.add_argument('--train_ldmk', type=str, default='train_landmark_R50', help='')
    parser.add_argument('--val_ldmk', type=str, default='val_landmark_R50', help='')
    args = parser.parse_args()
    print(args)
    train_test()
