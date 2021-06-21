import math

import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch import nn
from torch.nn import Parameter
import pdb
import numpy as np


def DBO(x,sigma=1):
    b,c,h,w = x.shape
    x=x.permute(0,2,3,1)
    y=x.unsqueeze(4).expand(-1,-1,-1,-1,c)
    z = x.repeat(1,1,1,c).view(b,h,w,c,c)
    delta = torch.abs(y-z)
    g=torch.exp(-(delta**2)/(sigma**2))
    term = g*y
    out = term.sum(3)/g.sum(3)
    return out.permute(0,3,1,2)


 
class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, dilation=self.conv.dilation, groups=self.conv.groups)

            return out_normal - self.theta * out_diff


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
  
    def forward(self, x):
        return self.conv(x)

    
        
class SpatialAttention(nn.Module):
    def __init__(self, kernel = 3):
        super(SpatialAttention, self).__init__()


        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel, padding=kernel//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        
        return self.sigmoid(x)


class CDCLowCell(nn.Module):
    def __init__(self, basic_conv, theta):
        super(CDCLowCell, self).__init__()
        self.Block1 = nn.Sequential(
           basic_conv(80, 160, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(160),
            nn.ReLU(),  
            
            basic_conv(160, int(160*1.6), kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(int(160*1.6)),
            nn.ReLU(),  
            basic_conv(int(160*1.6), 160, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(160),
            nn.ReLU(), 
        )
    def forward(self, x):
        return self.Block1(x)
    

class CDCMidCell(nn.Module):
    def __init__(self, basic_conv, theta):
        super(CDCMidCell, self).__init__()
        self.Block2 = nn.Sequential(
            basic_conv(160, int(160*1.2), kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(int(160*1.2)),
            nn.ReLU(),  
            basic_conv(int(160*1.2), 160, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(160),
            nn.ReLU(),  
            basic_conv(160, int(160*1.4), kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(int(160*1.4)),
            nn.ReLU(),  
            basic_conv(int(160*1.4), 160, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(160),
            nn.ReLU(),  
        )
    def forward(self, x):
        return self.Block2(x)
    
    
class CDCHighCell(nn.Module):
    def __init__(self, basic_conv, theta):
        super(CDCHighCell, self).__init__()
        self.Block3 = nn.Sequential(
            basic_conv(160, 160, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(160),
            nn.ReLU(), 
            basic_conv(160, int(160*1.2), kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(int(160*1.2)),
            nn.ReLU(),  
            basic_conv(int(160*1.2), 160, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(160),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.Block3(x)


    

class BCNpp(nn.Module):

    def __init__(self, basic_conv=Conv2d_cd, theta=0.7, map_size=(32,32), dbo_sigma = 1):   
        super(BCNpp, self).__init__()
        
        self.sigma = dbo_sigma
        self.conv1 = nn.Sequential(
            basic_conv(3, 80, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(80),
            nn.ReLU(),  
        )
        self.Block1 = CDCLowCell(basic_conv, theta)
        self.Block1_DBO = CDCLowCell(basic_conv, theta)
        self.Pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.Block2 = CDCMidCell(basic_conv, theta)
        self.Block2_DBO = CDCMidCell(basic_conv, theta)
        self.Pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.Block3 = CDCHighCell(basic_conv, theta)
        self.Block3_DBO = CDCHighCell(basic_conv, theta)
        self.Pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Original
        
        self.lastconv1 = nn.Sequential(
            basic_conv(160*3, 160, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(160),
            nn.ReLU(),
            basic_conv(160, 1, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.ReLU(),     
        )
        self.sa1 = SpatialAttention(kernel = 7)
        self.sa2 = SpatialAttention(kernel = 5)
        self.sa3 = SpatialAttention(kernel = 3)
        self.downsample32x32 = nn.Upsample(size=map_size, mode='bilinear',align_corners=False)

 
    def forward(self, x):	    	# x [3, 256, 256]
        
        x_input = x
        x = self.conv1(x)		   
        
        x_Res1 = self.Block1(x)
        x_Base1 = self.Block1_DBO(DBO(x, self.sigma))
        x_Block1 = self.Pool1(x_Res1 + x_Base1)
        x_Res1 = self.Pool1(x_Res1)
        
        attention1 = self.sa1(x_Block1)
        x_Block1_SA = attention1 * x_Block1
        x_Block1_32x32 = self.downsample32x32(x_Block1_SA)   
        
        x_Res2 = self.Block2(x_Res1)
        x_Base2 = self.Block2_DBO(DBO(x_Res1, self.sigma))
        x_Block2 = self.Pool2(x_Res2 + x_Base2)
        x_Res2 = self.Pool2(x_Res2)
        
        attention2 = self.sa2(x_Block2)  
        x_Block2_SA = attention2 * x_Block2
        x_Block2_32x32 = self.downsample32x32(x_Block2_SA)  
        
        x_Res3 = self.Block3(x_Res2)
        x_Base3 = self.Block3_DBO(DBO(x_Res2, self.sigma))
        x_Block3 = self.Pool3(x_Res3 + x_Base3)
        
        
        attention3 = self.sa3(x_Block3)  
        x_Block3_SA = attention3 * x_Block3	
        x_Block3_32x32 = self.downsample32x32(x_Block3_SA)   
        
        x_concat = torch.cat((x_Block1_32x32,x_Block2_32x32,x_Block3_32x32), dim=1)    
        
        #pdb.set_trace()
        
        map_x = self.lastconv1(x_concat)
        
        map_x = map_x.squeeze(1)
        
        return map_x
        #return map_x, x_concat, attention1, attention2, attention3, x_input
		




