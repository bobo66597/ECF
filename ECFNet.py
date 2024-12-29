import torch
import torch.nn as nn

from torchvision import models
# from resnet import resnet34
# import resnet
from torch.nn import functional as F
import numpy as np
from typing import Any, Callable

from torch import Tensor

from convLayer import Baseconv2d 
from poolLayer import AdaptiveAvgPool2d
from utils import shuffleTensor, setMethod, callMethod, makeDivisible, local_shuffle_tensor, get_target_centers, adjust_coordinates, get_datasets_info_with_keys, read_color_array, read_gray_array
import albumentations as A
import cv2

import random

class VGG_CBAM_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        outt = self.ca(out) * out
        out = self.sa(out) * out
        out = self.relu(out)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Res_CBAM_block(nn.Module):    #整合空间和通道注意力机制
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Res_CBAM_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class middelmode(nn.Module):  #
    def __init__(self, base_channel, stride = 1):     #inch=16   bc = 16
        super(middelmode, self).__init__()
        self.conv = nn.Conv2d(1, base_channel, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(base_channel)
        self.relu = nn.ReLU(inplace = True)
        self.bn2 = nn.BatchNorm2d(base_channel)
        self.conv1 =  nn.Conv2d(base_channel, 1, kernel_size = 1, stride = stride)
        self.convxh =  nn.Conv2d(1, base_channel, kernel_size = 1, stride = stride)
        self.convmx =  nn.Conv2d(base_channel*2, base_channel*4, kernel_size = 1, stride = stride)
        self.convmx_out =  nn.Conv2d(base_channel*2, base_channel, kernel_size = 1, stride = stride)
        self.dconv = nn.Conv2d(1,base_channel, kernel_size = 3,stride = stride, padding=2,dilation=2)
        self.cbam = Res_CBAM_block(base_channel*7,base_channel*2)   #不一定是三倍记得改
        self.pool  = nn.MaxPool2d(2, 2)
        self.channel = ChannelAttention(base_channel*4)
        self.up    = nn.Upsample(scale_factor=2,   mode='bilinear', align_corners=True)

    def forward(self, xh, x, xl):        #xh16,x32,xl64
        
        xh = self.conv1(xh)         #c,ich->1      xh256
        xh1 = self.relu(self.bn1(self.conv(xh)))     #1->outchi      xh1  256
        xh2 =self.relu(self.bn1(self.dconv(xh)))     #1->outchi      xh2  256
        mx = torch.cat([self.pool(self.convxh(xh)),x,self.up(xl)],dim=1)     #维度统一128     mx128*128
        xl1 = self.channel(xl)                            #xl1->inchi

        mx_out = self.cbam(mx)                           #mx_out->outchi128*128
        xh_out =self.relu((xh1+xh2)*xh+self.up(self.convmx_out(mx_out)))    #（xh1+xh2）*xh->outchi256*256    mx_out->outchi128*128    xh_out->outchi256*256
        xl_out = self.relu(xl1*xl+self.pool(self.convmx(mx_out)))               #xl1*xl->inchi64*64     xl_out->inchi64*64

        return xh_out, mx_out, xl_out       #xh_out->256*256,16, mx_out->128*128,32, xl_out ->64*64,64

     

class EdgeSobelAttention(nn.Module):
    def __init__(self, in_channels):
        super(EdgeSobelAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

        # Sobel 算子
        self.sobel_x = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)

        # 初始化 Sobel 卷积核
        sobel_kernel_x = torch.tensor([ [1, 0, -1], 
                                        [2, 0, -2], 
                                        [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_kernel_y = torch.tensor([ [1, 2, 1], 
                                        [0, 0, 0], 
                                        [-1,-2,-1]], dtype=torch.float32).view(1, 1, 3, 3)

        # 将 Sobel 卷积核复制到所有输入通道
        self.sobel_x.weight = nn.Parameter(sobel_kernel_x.repeat(1, in_channels, 1, 1), requires_grad=False)
        self.sobel_y.weight = nn.Parameter(sobel_kernel_y.repeat(1, in_channels, 1, 1), requires_grad=False)

    def forward(self, x):
        # 使用 Sobel 算子提取边缘特征
        edges_x = self.sobel_x(x)
        edges_y = self.sobel_y(x)
        edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)

        # 生成注意力权重
        attention_weights = self.sigmoid(edges)

        # 生成加权后的输出特征图
        out = x * attention_weights
        #out = cv2.bilateralFilter(out, d=4, sigmaColor=20, sigmaSpace=20)
        
        return out




class SpatialAttentionc(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionc, self).__init__()
        padding = 3 if kernel_size == 7 else (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class HierarchicalAttention(nn.Module):
    def __init__(self, in_channels,out_channels,stride = 1):
        super(HierarchicalAttention, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None
        self.ca_low = ChannelAttention(out_channels)  
        self.sa_low = SpatialAttentionc(kernel_size=3)  

        self.ca_mid = ChannelAttention(out_channels)  
        self.sa_mid = SpatialAttentionc(kernel_size=5)  

        self.ca_high = ChannelAttention(out_channels)  
        self.sa_high = SpatialAttentionc(kernel_size=7)  

        self.ea = EdgeSobelAttention(out_channels)

        self.adaptive_weight = nn.Parameter(torch.ones(3), requires_grad=True)

    def forward(self, x):

        if self.shortcut is not None:
            residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ea(out) * out

        x_low = self.ca_low(out) * out
        x_low = self.sa_low(x_low) * x_low


        x_mid = self.ca_mid(out) * out
        x_mid = self.sa_mid(x_mid) * x_mid


        x_high = self.ca_high(out) * out
        x_high = self.sa_high(x_high) * x_high


        weights = torch.softmax(self.adaptive_weight, dim=0)
        out = weights[0] * x_low + weights[1] * x_mid + weights[2] * x_high
        out += residual
        out = self.relu(out)

        return out



class Res_ERCAM_block(nn.Module):   
    def __init__(self, in_channels, out_channels,in_feature, out_feature, stride = 1):
        super(Res_ERCAM_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ea = EdgeSobelAttention(out_channels)
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ea(out) * out
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out


     #动态尺度融合模块

class MoCAttention3(nn.Module):
    def __init__(self, InChannels: int, HidChannels: int, stage: int = int, stage1: int = int, SqueezeFactor: int = 4,
                 PoolRes: list = [1, 2, 4, 8], Act: Callable[..., nn.Module] = nn.ReLU,
                 ScaleAct: Callable[..., nn.Module] = nn.Sigmoid, MoCOrder: bool = True, **kwargs: Any) -> None:
        super().__init__()
        if HidChannels is None:
            HidChannels = max(InChannels // SqueezeFactor, 32)

        AllPoolRes = PoolRes + [1] if 1 not in PoolRes else PoolRes
        for k in AllPoolRes:
            Pooling = nn.AdaptiveAvgPool2d(k)
            setattr(self, f'Pool{k}', Pooling)

        self.SELayer = nn.Sequential(
            nn.Conv2d(InChannels, HidChannels, 1),
            Act(),
            nn.Conv2d(HidChannels, InChannels, 1),
            ScaleAct(),
        )
        self.conv = nn.Conv2d(InChannels, HidChannels, kernel_size = 1, stride = 1)
        self.PoolRes = PoolRes
        self.MoCOrder = MoCOrder
        self.stage = stage
        self.stage1 = stage1

    def monteCarloSample(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        
 
        scale_factor = self.stage1  
        centers = get_target_centers(mask)
        adjusted_centers = adjust_coordinates(centers, scale_factor)  

        if self.training:
            x1 = local_shuffle_tensor(x, adjusted_centers, target_size=32) if self.MoCOrder else x
            AttnMap: torch.Tensor = getattr(self, f'Pool{self.stage}')(x1)
            if AttnMap.shape[-1] > 1:
                AttnMap = AttnMap.flatten(2)
                AttnMap = AttnMap[:, :, torch.randperm(AttnMap.shape[-1])[0]]
                AttnMap = AttnMap[:, :, None, None]  # squeeze twice
        else:
            AttnMap: torch.Tensor = getattr(self, 'Pool1')(x)
        
        return   AttnMap

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        AttnMap = self.monteCarloSample(x, mask)
        x = self.conv( x * self.SELayer(AttnMap))
        return x
    

class ECFNet(nn.Module):
    def __init__(self, num_classes, input_channels, block, num_blocks, nb_filter,deep_supervision=True):
        super(ECFNet, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        self.deep_supervision = deep_supervision

        self.conv0_3_final = self._make_layer(block, nb_filter[0]*4, nb_filter[0],0,0,num_blocks[0])

        self.pool  = nn.MaxPool2d(2, 2)
        self.up    = nn.Upsample(scale_factor=2,   mode='bilinear', align_corners=True)
        self.down  = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

        self.up_4  = nn.Upsample(scale_factor=4,   mode='bilinear', align_corners=True)
        self.up_8  = nn.Upsample(scale_factor=8,   mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=16,  mode='bilinear', align_corners=True)

        self.mc0_0 = MoCAttention3(nb_filter[0]*2,nb_filter[0],8,1)
        self.mc1_0 = MoCAttention3(nb_filter[1]*2,nb_filter[1],4,2)
        self.mc2_0 = MoCAttention3(nb_filter[2]*2,nb_filter[2],2,4)
        self.mc3_0 = MoCAttention3(nb_filter[3]*2,nb_filter[3],1,8)


        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0],0,0, num_blocks[0])   #block=Res_CBAM_block
        self.conv0_0up = self._make_layer(HierarchicalAttention, input_channels, nb_filter[0],pow(512, 2),512,num_blocks = 1)


        self.conv1_0 = self._make_layer(block, nb_filter[0],  nb_filter[1],0,0, num_blocks[1])
        self.conv1_0up = self._make_layer(HierarchicalAttention, nb_filter[0], nb_filter[1],pow(256, 2), 256,num_blocks = 1)

        self.conv2_0 = self._make_layer(block, nb_filter[1],  nb_filter[2],0,0, num_blocks[2])
        self.conv2_0up = self._make_layer(HierarchicalAttention, nb_filter[1],  nb_filter[2],pow(128, 2), 128,num_blocks = 1)

        self.conv3_0 = self._make_layer(block, nb_filter[2],  nb_filter[3],0,0, num_blocks[3])
        self.conv3_0up = self._make_layer(HierarchicalAttention, nb_filter[2],  nb_filter[3],pow(64 ,2), 64,num_blocks = 1 )

        self.convmid = middelmode(nb_filter[0])


        self.conv0_1 = self._make_layer(block, nb_filter[0] + nb_filter[1],  nb_filter[0],0,0,num_blocks[0])
        self.conv1_1 = self._make_layer(block, nb_filter[1] + nb_filter[2] + nb_filter[0],  nb_filter[1],0,0, num_blocks[0])
        self.conv2_1 = self._make_layer(block, nb_filter[2] + nb_filter[3] + nb_filter[2],  nb_filter[2],0,0, num_blocks[1])


        self.conv0_2 = self._make_layer(block, nb_filter[0]*3, nb_filter[0],0,0,num_blocks[0])
        self.conv1_2 = self._make_layer(block, nb_filter[1]*2 + nb_filter[0] + nb_filter[2], nb_filter[1],0,0, num_blocks[0])


        self.conv0_3 = self._make_layer(block, nb_filter[0]*3 + nb_filter[1], nb_filter[0],0,0,num_blocks[0])


        self.conv3_0_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        self.conv2_1_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        self.conv1_2_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)


        if self.deep_supervision:
            self.final1 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final  = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, input_channels,  output_channels, in_feature, out_feature, num_blocks=1):
        layers = []
        if block == Res_ERCAM_block :
           layers.append(block(input_channels, output_channels, in_feature, out_feature))
           for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels, in_feature, out_feature))
           return nn.Sequential(*layers)
    
        else:
           layers.append(block(input_channels, output_channels))
           for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
           return nn.Sequential(*layers)

    def forward(self, data, data2, labels):   #data0

        x0_0 = self.conv0_0(data)           #x0_0256
        x0_0up = self.conv0_0up(data2)     #x0_0up 512
        x0_0M = self.mc0_0(torch.cat([x0_0,self.pool(x0_0up)],1),labels)  #mc
        x1_0 = self.conv1_0(self.pool(x0_0))   #x1_0128
        x1_0up = self.conv1_0up(self.pool(x0_0up)) #x1_0up256
        x1_0M = self.mc1_0(torch.cat([x1_0,self.pool(x1_0up)],1),labels)   #mc
        x2_0 = self.conv2_0(self.pool(x1_0))     #x2_064
        x2_0up = self.conv2_0up(self.pool(x1_0up))           #x2_0up128
        x2_0M = self.mc2_0(torch.cat([x2_0,self.pool(x2_0up)],1),labels)  #mc
        x3_0 = self.conv3_0(self.pool(x2_0))        #x3_0 32
        x3_0up = self.conv3_0up(self.pool(x2_0up))      #x3_0up 64
        x3_0M = self.mc3_0(torch.cat([x3_0,self.pool(x3_0up)],1),labels)
        x0_1 = self.conv0_1(torch.cat([x0_0M, self.up(x1_0M)], 1)) ################################
        x1_1h,x1_1m,x1_1l = self.convmid(x0_1,x1_0M,x2_0M)   #x00m16,x10m32,x20m64
        x0_2 = self.conv0_2(torch.cat([x0_0M, x0_1, x1_1h], 1))  #########################################

        x2_1 = self.conv2_1(torch.cat([x2_0M, self.up(x3_0M),x1_1l], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0M, x1_1m, self.pool(x0_2), self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0M, x0_2, x0_1, self.up(x1_2)], 1))



        Final_x0_3 = self.conv0_3_final(
            torch.cat([self.up_8(self.conv3_0_1x1(x3_0)),
                       self.up_4(self.conv2_1_1x1(x2_1)),self.up(self.conv1_2_1x1(x1_2)), x0_3], 1))

        if self.deep_supervision:
            output1 = self.final1(self.up_8(self.conv3_0_1x1(x3_0)))
            output2 = self.final2(self.up_4(self.conv2_1_1x1(x2_1)))      #self.final2(self.up_4(self.conv2_1_1x1(x2_1)))
            output3 = self.final3(self.up(self.conv1_2_1x1(x1_2)))
            output4 = self.final4(Final_x0_3)
            return  [output1, output2, output3, output4]   #@ if make heatmap: Final_x0_3          if normal  [output1, output2, output3, output4]
        else:
            output = self.final(Final_x0_3)
            return output

