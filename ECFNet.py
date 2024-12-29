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
from base_dataset import _BaseSODDataset
from dysample import DySample
from einops import rearrange

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


class VGG_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        #self.ca = ChannelAttention(out_channels)
        #self.sa = SpatialAttention()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
       #ut = self.ca(out) * out
       #out = self.sa(out) * out
        out = self.relu(out)
        return out


class DYup_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dyup = nn.Upsample(scale_factor=2,   mode='bilinear', align_corners=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        #self.ca = ChannelAttention(out_channels)
        #self.sa = SpatialAttention()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dyup(out)
       #ut = self.ca(out) * out
       #out = self.sa(out) * out
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


from einops import rearrange

class GlobalRegionSelfAttention(nn.Module):
    def __init__(self, in_channels, in_feature, out_feature):
        super(GlobalRegionSelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, out_channels=1, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, out_channels=1, kernel_size=1)
        self.query_line = nn.Linear(in_features=in_feature, out_features=out_feature)
        self.key_line = nn.Linear(in_features=in_feature, out_features=out_feature)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.s_conv = nn.Conv2d(in_channels=1, out_channels=in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        batch_size, C, H, W = x.size()
        #region_h = H // self.region_size
        #region_w = W // self.region_size
        region_size = H // 2
        # 提取全图特征
        #query_global = self.query_conv(x).view(batch_size, -1, H * W)
        #key_global = self.key_conv(x).view(batch_size, -1, H * W)


        q = rearrange(self.query_conv(x), 'b 1 h w -> b (h w)')
        k = rearrange(self.key_conv(x), 'b 1 h w -> b (h w)')
        query_global = rearrange(self.query_line(q), 'b h -> b h 1')
        key_global = rearrange(self.key_line(k), 'b h -> b 1 h')
        att = rearrange(torch.matmul(query_global, key_global), 'b h w -> b 1 h w')
        att = self.softmax(self.s_conv(att))
        #att = self.pool(att)

        # 使用全图的 query 和 key 计算注意力权重
       #attention_scores = torch.bmm(query_global.permute(0, 2, 1), key_global)
        #attention_map = self.softmax(attention_scores)  # 计算全局注意力权重
       
        regions = []
        outs = []
        #region_index = 0
        for i in range(0, H, region_size):  # 遍历高度
            for j in range(0, W, region_size):  # 遍历宽度
        # 提取区域
             region = x[:, :, i:i + region_size, j:j + region_size]
        
        # 将该区域命名为 'region_1', 'region_2', ..., 'region_16'
             #region_name = f'region_{region_index}'
        
        # 将命名的区域存储到数组中
             regions.append(region)    #region_name
        
        # 更新区域索引
             #region_index += 1

            # 打印输出所有区域的名称和形状
        #for name, region in regions:
             #print(f"{name}: 形状为 {region.shape}")
        for v in regions:
            out  = torch.matmul(att, v)
            outs.append(out)
        # 提取区域的 value 特征

        row1 = torch.cat([outs[0], outs[1]], dim=3)  # 拼接第1行 512/256=2
        row2 = torch.cat([outs[2], outs[2]], dim=3)  # 拼接第2行 (region_5 to region_8)
        #row3 = torch.cat(regions[:12], dim=-1)  # 拼接第3行 (region_9 to region_12)
        #row4 = torch.cat(regions[12:16], dim=-1)  # 拼接第4行 (region_13 to region_16) 
        out = torch.cat([row1, row2], dim=2)

        #value = self.value_conv(x).view(batch_size, -1, region_h, self.region_size, region_w, self.region_size)

        # 将全局注意力权重应用于每个区域的 value
        #out = torch.einsum('bchw,bcrhw->bcrhws', att, value)

        # 调整输出形状
        #out = out.view(batch_size, C, H, W)
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


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1 / 2 * (x_out11 + x_out21) * x_out
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)
        return x_out
    

class TripletAttentionblock(nn.Module):    #整合空间和通道注意力机制
    def __init__(self, in_channels, out_channels, stride = 1):
        super(TripletAttentionblock, self).__init__()
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
        self.tatt = TripletAttention()

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
        out = self.tatt(out)
        return out


# 输入 B C H W,  输出 B C H W
if __name__ == '__main__':
    input = torch.randn(3, 32, 64, 64)
    triplet = TripletAttention()
    output = triplet(input)
    print(output.shape)









class GlobalRegionSelfAttention1(nn.Module):
    def __init__(self, in_channels, in_feature, out_feature):
        super(GlobalRegionSelfAttention1, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, out_channels=1, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, out_channels=1, kernel_size=1)
        self.query_line = nn.Linear(in_features=in_feature, out_features=out_feature)
        self.key_line = nn.Linear(in_features=in_feature, out_features=out_feature)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.s_conv = nn.Conv2d(in_channels=1, out_channels=in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        batch_size, C, H, W = x.size()
        #region_h = H // self.region_size
        #region_w = W // self.region_size
        region_size = H // 2
        # 提取全图特征
        #query_global = self.query_conv(x).view(batch_size, -1, H * W)
        #key_global = self.key_conv(x).view(batch_size, -1, H * W)


        q = rearrange(self.query_conv(x), 'b 1 h w -> b (h w)')
        k = rearrange(self.key_conv(x), 'b 1 h w -> b (h w)')
        query_global = rearrange(self.query_line(q), 'b h -> b h 1')
        key_global = rearrange(self.key_line(k), 'b h -> b 1 h')
        att = rearrange(torch.matmul(query_global, key_global), 'b h w -> b 1 h w')
        att = self.softmax(self.s_conv(att))
        #att = self.pool(att)
        #out  = torch.matmul(att, x)

        #value = self.value_conv(x).view(batch_size, -1, region_h, self.region_size, region_w, self.region_size)

        # 将全局注意力权重应用于每个区域的 value
        #out = torch.einsum('bchw,bcrhw->bcrhws', att, value)

        # 调整输出形状
        #out = out.view(batch_size, C, H, W)
        return att

class middelmode(nn.Module):  #中央整合处理层
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
        self.ca_low = ChannelAttention(out_channels)  # 低层次通道注意力
        self.sa_low = SpatialAttentionc(kernel_size=3)  # 低层次空间注意力

        self.ca_mid = ChannelAttention(out_channels)  # 中层次通道注意力
        self.sa_mid = SpatialAttentionc(kernel_size=5)  # 中层次空间注意力

        self.ca_high = ChannelAttention(out_channels)  # 高层次通道注意力
        self.sa_high = SpatialAttentionc(kernel_size=7)  # 高层次空间注意力

        self.ea = EdgeSobelAttention(out_channels)

        # 自适应加权模块
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
        # 低层次注意力
        x_low = self.ca_low(out) * out
        x_low = self.sa_low(x_low) * x_low

        # 中层次注意力
        x_mid = self.ca_mid(out) * out
        x_mid = self.sa_mid(x_mid) * x_mid

        # 高层次注意力
        x_high = self.ca_high(out) * out
        x_high = self.sa_high(x_high) * x_high

        # 自适应加权和融合
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
        #self.ra = GlobalRegionSelfAttention1(out_channels,in_feature, out_feature)   #self.ra = GlobalRegionSelfAttention(out_channels,in_feature, out_feature)
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
       # out  = torch.matmul(att, v)
        out += residual
        out = self.relu(out)
        return out





#*****************************************************************************************************#








     #动态尺度融合模块

class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = nn.BatchNorm2d(out_planes)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x



class DSC(nn.Module):
    def __init__(self, in_channels):
        super(DSC, self).__init__()
        self.conv3x3=nn.Conv2d(in_channels=in_channels, out_channels=in_channels,dilation=1,kernel_size=3, padding=1)
        
        self.bn=nn.ModuleList([nn.BatchNorm2d(in_channels),nn.BatchNorm2d(in_channels),nn.BatchNorm2d(in_channels)]) 
        self.conv1x1=nn.ModuleList([nn.Conv2d(in_channels=2*in_channels, out_channels=in_channels,dilation=1,kernel_size=1, padding=0),
                                    nn.Conv2d(in_channels=2*in_channels, out_channels=in_channels,dilation=1,kernel_size=1, padding=0)])
        self.conv3x3_1=nn.ModuleList([nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2,dilation=1,kernel_size=3, padding=1),
                                      nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2,dilation=1,kernel_size=3, padding=1)])
        self.conv3x3_2=nn.ModuleList([nn.Conv2d(in_channels=in_channels//2, out_channels=2,dilation=1,kernel_size=3, padding=1),
                                      nn.Conv2d(in_channels=in_channels//2, out_channels=2,dilation=1,kernel_size=3, padding=1)])
        self.conv_last=ConvBnRelu(in_planes=in_channels,out_planes=in_channels,ksize=1,stride=1,pad=0,dilation=1)
        self.norm = nn.Sigmoid()
        self.conv1= nn.Conv2d(in_channels*2, 1, kernel_size=1, padding=0)
        self.dconv1=nn.Conv2d(in_channels*2, in_channels, kernel_size=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))
    
        self.relu=nn.ReLU(inplace=True)

    def forward(self, x):

        x_size= x.size()


          #对输入图片进行不同扩张程度的卷积（1，2，4）
        branches_1=self.conv3x3(x)                                                      #1
        branches_1=self.bn[0](branches_1)

        branches_2=F.conv2d(x,self.conv3x3.weight,padding=2,dilation=2)#share weight     2
        branches_2=self.bn[1](branches_2)

        branches_3=F.conv2d(x,self.conv3x3.weight,padding=4,dilation=4)#share weight     4 
        branches_3=self.bn[2](branches_3)


            # 处理扩张程度1和2的特征图


        feat=torch.cat([branches_1,branches_2],dim=1) 

        feat_g =feat
        # print(feat_g.shape)
        feat_g1 = self.relu(self.conv1(feat_g))
        feat_g1 = self.norm(feat_g1)
        
        out1 = feat_g * feat_g1
        out1 = self.dconv1(out1)
       

        # feat=feat_cat.detach()
        feat=self.relu(self.conv1x1[0](feat))
        feat=self.relu(self.conv3x3_1[0](feat))
        att=self.conv3x3_2[0](feat)
        att = F.softmax(att, dim=1)
        
        att_1=att[:,0,:,:].unsqueeze(1)#分离
        att_2=att[:,1,:,:].unsqueeze(1)

        fusion_1_2=att_1*branches_1+att_2*branches_2 +out1



          
       # 处理扩张程度12和4的特征图


        feat1=torch.cat([fusion_1_2,branches_3],dim=1)

        feat_g =feat1

        feat_g1 = self.relu(self.conv1(feat_g))
        feat_g1 = self.norm(feat_g1)

        out2 = feat_g * feat_g1
        out2 = self.dconv1(out2)


        # feat=feat_cat.detach()
        feat1=self.relu(self.conv1x1[0](feat1))
        feat1=self.relu(self.conv3x3_1[0](feat1))
        att1=self.conv3x3_2[0](feat1)
        att1 = F.softmax(att1, dim=1)
      
        
        att_1_2=att1[:,0,:,:].unsqueeze(1)#分离
        att_3=att1[:,1,:,:].unsqueeze(1)
       

        ax=self.relu(self.gamma*(att_1_2*fusion_1_2+att_3*branches_3 +out2)+(1-self.gamma)*x)
        ax=self.conv_last(ax)

        return ax






'''class MoCAttention(nn.Module):
    # Monte carlo attention
    def __init__(
        self,
        InChannels: int,
        HidChannels: int,
        SqueezeFactor: int=4,
        PoolRes: list=[1, 2, 3],
        Act: Callable[..., nn.Module]=nn.ReLU,
        ScaleAct: Callable[..., nn.Module]=nn.Sigmoid,
        MoCOrder: bool=True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if HidChannels is None:
            HidChannels = max(makeDivisible(InChannels // SqueezeFactor, 8), 32)
        
        AllPoolRes = PoolRes + [1] if 1 not in PoolRes else PoolRes
        for k in AllPoolRes:
            Pooling = AdaptiveAvgPool2d(k)
            setMethod(self, 'Pool%d' % k, Pooling)
            
        self.SELayer = nn.Sequential(
            Baseconv2d(InChannels, HidChannels, 1, ActLayer=Act),
            Baseconv2d(HidChannels, InChannels, 1, ActLayer=ScaleAct),
        )
        
        self.PoolRes = PoolRes
        self.MoCOrder = MoCOrder
        
    def monteCarloSample(self, x: Tensor) -> Tensor:
        if self.training:
            PoolKeep = np.random.choice(self.PoolRes)   #从 self.PoolRes 中随机选择一个池化分辨率 PoolKeep
            x1 = shuffleTensor(x)[0] if self.MoCOrder else x   #如果 self.MoCOrder 为 True，则调用 shuffleTensor 函数打乱张量 x 的顺序
            AttnMap: Tensor = callMethod(self, 'Pool%d' % PoolKeep)(x1)
            if AttnMap.shape[-1] > 1:
                AttnMap = AttnMap.flatten(2)
                AttnMap = AttnMap[:, :, torch.randperm(AttnMap.shape[-1])[0]]
                AttnMap = AttnMap[:, :, None, None] # squeeze twice
        else:
            AttnMap: Tensor = callMethod(self, 'Pool%d' % 1)(x)
            
        return AttnMap
        
    def forward(self, x: Tensor) -> Tensor:
        AttnMap = self.monteCarloSample(x)
        return x * self.SELayer(AttnMap)'''


#mcatt_change#############################################################################################


'''class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_channels):
        super(PixelShuffleUpsample, self).__init__()
        
        self.pixel_shuffle1 = nn.PixelShuffle(6)
        self.pixel_shuffle2 = nn.PixelShuffle(3)
        self.pixel_shuffle3 = nn.PixelShuffle(2)
        
       
            
        self.conv6 = nn.Conv2d(in_channels, 36, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, 4, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, 9, kernel_size=1)
        self.conv6_c = nn.Conv2d(36 // (6**2),in_channels,kernel_size=1)
        self.conv3_c = nn.Conv2d(4 // (2**2),in_channels,kernel_size=1)
        self.conv2_c = nn.Conv2d(9 // (3**2),in_channels,kernel_size=1)


    def forward(self, x):
        # 输入尺寸 (B, C, H, W)
        # 使用 PixelShuffle 进行上采样
        #=[1,2,3]
        if x.size(3) == 1:   
            x = self.conv6(x)
            x = self.pixel_shuffle1(x)
            x = self.conv6_c(x) 
        elif x.size(3) == 3:
            x = self.conv3(x)
            x = self.pixel_shuffle3(x)
            x = self.conv3_c(x) 
        else :
            x = self.conv2(x)
            x = self.pixel_shuffle2(x)  
            x = self.conv2_c(x) # 输出尺寸 (B, C // (upscale_factor**2), H * upscale_factor, W * upscale_factor)
        
        return x'''


class MoCAttention2(nn.Module):
    # Monte carlo attention
    def __init__(
        self,
        InChannels: int,
        HidChannels: int,
        stage = int,
        SqueezeFactor: int=4,
        PoolRes:list = [1,2,4,8],
        
        Act: Callable[..., nn.Module]=nn.ReLU,
        ScaleAct: Callable[..., nn.Module]=nn.Sigmoid,
        MoCOrder: bool=True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if HidChannels is None:
            HidChannels = max(makeDivisible(InChannels // SqueezeFactor, 8), 32)
        
        AllPoolRes = PoolRes + [1] if 1 not in PoolRes else PoolRes
        for k in AllPoolRes:
            Pooling = AdaptiveAvgPool2d(k)
            setMethod(self, 'Pool%d' % k, Pooling)
            
        self.SELayer = nn.Sequential(
            Baseconv2d(InChannels, HidChannels, 1, ActLayer=Act),
            Baseconv2d(HidChannels, InChannels, 1, ActLayer=ScaleAct),
        )
        
        self.PoolRes = PoolRes
        self.MoCOrder = MoCOrder
        self.stage = stage
        
    def monteCarloSample(self, x: Tensor) -> Tensor:
        if self.training: 
            x1 = shuffleTensor(x)[0] if self.MoCOrder else x   #如果 self.MoCOrder 为 True，则调用 shuffleTensor 函数打乱张量 x 的顺序
            AttnMap: Tensor = callMethod(self, 'Pool%d' % self.stage)(x1)
            if AttnMap.shape[-1] > 1:
                AttnMap = AttnMap.flatten(2)
                AttnMap = AttnMap[:, :, torch.randperm(AttnMap.shape[-1])[0]]
                AttnMap = AttnMap[:, :, None, None] # squeeze twice
        else:
            AttnMap: Tensor = callMethod(self, 'Pool%d' % 1)(x)
            
        return AttnMap
        
    def forward(self, x: Tensor) -> Tensor:
        AttnMap = self.monteCarloSample(x)
        return x * self.SELayer(AttnMap)





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
        

 
        scale_factor = self.stage1  # 计算当前阶段的下采样比例
        centers = get_target_centers(mask)
        adjusted_centers = adjust_coordinates(centers, scale_factor)  # 调整坐标

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
    
##################################################################################################################

class UniRotate(A.DualTransform):
    """UniRotate the input by an angle selected randomly from the uniform distribution.

    Args:
        limit ((int, int) or int): range from which a random angle is picked. If limit is a single int
            an angle is picked from (-limit, limit). Default: (-90, 90)
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        border_mode (OpenCV flag): flag that is used to specify the pixel extrapolation method. Should be one of:
            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
            Default: cv2.BORDER_REFLECT_101
        value (int, float, list of ints, list of float): padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int, float,
                    list of ints,
                    list of float): padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        limit=90,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        value=None,
        mask_value=None,
        always_apply=False,
        p=0.5,
    ):
        super(UniRotate, self).__init__(always_apply, p)
        self.limit = A.to_tuple(limit)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def apply(self, img, angle=0, interpolation=cv2.INTER_LINEAR, **params):
        return A.rotate(img, angle, interpolation, self.border_mode, self.value)

    def apply_to_mask(self, img, angle=0, interpolation=cv2.INTER_LINEAR, **params):
        return A.rotate(img, angle, interpolation, self.border_mode, self.mask_value)

    def get_params(self):
        return {"angle": random.uniform(self.limit[0], self.limit[1])}

    def apply_to_bbox(self, bbox, angle=0, **params):
        return A.bbox_rotate(bbox, angle, params["rows"], params["cols"])

    def apply_to_keypoint(self, keypoint, angle=0, **params):
        return A.keypoint_rotate(keypoint, angle, **params)

    def get_transform_init_args_names(self):
        return ("limit", "interpolation", "border_mode", "value", "mask_value")




def ms_resize(img, scales, base_h=None, base_w=None, interpolation=cv2.INTER_LINEAR):
    assert isinstance(scales, (list, tuple))
    if base_h is None and base_w is None:
        h = img.shape[0]
        w = img.shape[1]
    else:
        h = base_h
        w = base_w
    return [A.resize(img, (h*0.5, w*0.5), interpolation=interpolation),A.resize(img, (h, w), interpolation=interpolation),A.resize(img, (h*2.0, w*2.0), interpolation=interpolation)]    #[A.resize(img, height=int(h * s), width=int(w * s), interpolation=interpolation) for s in scales]



def ss_resize(img, scale, base_h=None, base_w=None, interpolation=cv2.INTER_LINEAR):
    if base_h is None and base_w is None:
        h = img.shape[0]
        w = img.shape[1]
    else:
        h = base_h
        w = base_w
    return A.resize(img, height=int(h * scale), width=int(w * scale), interpolation=interpolation)





class rsBlock(nn.Module):
    def __init__(self):
        super().__init__()
        #image = read_color_array()
        #mask = read_gray_array(to_normalize=True, thr=0.5)
        self.base_shape = {"h": 256, "w": 256}

        #self.transformed = joint_trans(image=image, mask=mask)
        #image = transformed["image"]
       # mask = transformed["mask"]

        #base_h = self.base_shape["h"]
        #base_w = self.base_shape["w"]
        #images = ms_resize(image, scales=(0.5, 1.0, 1.5), base_h=base_h, base_w=base_w)
        #image_0_5 = torch.from_numpy(images[0]).permute(2, 0, 1)
        #image_1_0 = torch.from_numpy(images[1]).permute(2, 0, 1)
        #image_1_5 = torch.from_numpy(images[2]).permute(2, 0, 1)

        #mask = ss_resize(mask, scale=1.0, base_h=base_h, base_w=base_w)
        #mask_1_0 = torch.from_numpy(mask).unsqueeze(0)


        #joint_trans = A.Compose(
       #     [
       #         A.HorizontalFlip(p=0.5),
      #          UniRotate(limit=10, interpolation=cv2.INTER_LINEAR, p=0.5),
      #          A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
      #      ],
      #  )
       # self.reszie = A.Resize



    def forward(self, x):
        

        image = x
        tensor_image = image.cpu()
        numpy_image = tensor_image.permute(1, 2, 0).numpy()
        image = (numpy_image * 255).astype(np.uint8)
        print(type(image))

        #mask = read_gray_array(x1,to_normalize=True, thr=0.5)

        #trans = self.transformed(image=image, mask=mask)

        #image = trans["image"]
        #mask = trans["mask"]
        base_h = self.base_shape["h"]
        base_w = self.base_shape["w"]
        images = ms_resize(image, scales=(0.5, 1.0, 2.0), base_h=base_h, base_w=base_w)
        image_0_5 = torch.from_numpy(images[0]).permute(2, 0, 1)
        image_1_0 = torch.from_numpy(images[1]).permute(2, 0, 1)
        image_2_0 = torch.from_numpy(images[2]).permute(2, 0, 1)

        #mask = ss_resize(mask, scale=1.0, base_h=base_h, base_w=base_w)
        #mask_1_0 = torch.from_numpy(mask).unsqueeze(0)


        return image_0_5, image_1_0, image_2_0
            
        


#差分卷积

def conv_relu_bn(in_channel, out_channel, dirate=1):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=1,
            padding=dirate,
            dilation=dirate,
        ),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True),
    )




class CDC_conv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bias=True,
        kernel_size=3,
        padding=1,
        dilation=1,
        theta=0.7,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.theta = theta

    def forward(self, x):
        norm_out = self.conv(x)
        [c_out, c_in, kernel_size, kernel_size] = self.conv.weight.shape
        kernel_diff = self.conv.weight.sum(2).sum(2)
        kernel_diff = kernel_diff[:, :, None, None]
        diff_out = F.conv2d(
            input=x,
            weight=kernel_diff,
            bias=self.conv.bias,
            stride=self.conv.stride,
            padding=0,
        )
        out = norm_out - self.theta * diff_out
        return out



class new_conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(new_conv_block, self).__init__()
        self.conv_layer = nn.Sequential(
            conv_relu_bn(in_ch, in_ch, 1),
            conv_relu_bn(in_ch, out_ch, 1),
            conv_relu_bn(out_ch, out_ch, 1),
        )
        self.cdc_layer = nn.Sequential(
            CDC_conv(in_ch, out_ch), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
        self.dconv_layer = nn.Sequential(
            conv_relu_bn(in_ch, in_ch, 2),
            conv_relu_bn(in_ch, out_ch, 4),
            conv_relu_bn(out_ch, out_ch, 2),
        )
        self.final_layer = conv_relu_bn(out_ch * 3, out_ch, 1)

    def forward(self, x):
        conv_out = self.conv_layer(x)
        cdc_out = self.cdc_layer(x)
        dconv_out = self.dconv_layer(x)
        out = torch.concat([conv_out, cdc_out, dconv_out], dim=1)
        out = self.final_layer(out)
        return out





class DNANet(nn.Module):
    def __init__(self, num_classes, input_channels, block, num_blocks, nb_filter,deep_supervision=True):
        super(DNANet, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        self.deep_supervision = deep_supervision

        self.conv0_3_final = self._make_layer(block, nb_filter[0]*4, nb_filter[0],0,0,num_blocks[0])

        self.pool  = nn.MaxPool2d(2, 2)
        self.up    = nn.Upsample(scale_factor=2,   mode='bilinear', align_corners=True)
        self.down  = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

        self.up_4  = nn.Upsample(scale_factor=4,   mode='bilinear', align_corners=True)
        self.up_8  = nn.Upsample(scale_factor=8,   mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=16,  mode='bilinear', align_corners=True)

        #self.DSC=DSC(nb_filter[4]) 
        self.mc0_0 = MoCAttention3(nb_filter[0]*2,nb_filter[0],8,1)
        self.mc1_0 = MoCAttention3(nb_filter[1]*2,nb_filter[1],4,2)
        self.mc2_0 = MoCAttention3(nb_filter[2]*2,nb_filter[2],2,4)
        self.mc3_0 = MoCAttention3(nb_filter[3]*2,nb_filter[3],1,8)

       #self.rs = rsBlock()


        #新主干 



        '''self.nconv0_0 = new_conv_block(input_channels,nb_filter[0])  #3,16
        self.nconv1_0 = new_conv_block(nb_filter[0],nb_filter[1])    #16,32
        self.nconv2_0 = new_conv_block(nb_filter[1],nb_filter[2])    #32,64
        self.nconv3_0 = new_conv_block(nb_filter[2],nb_filter[3])    #64,128
        self.nconv4_0 = new_conv_block(nb_filter[3],nb_filter[4])   #128,256'''
        

 
        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0],0,0, num_blocks[0])   #block=Res_CBAM_block
        #self.conv0_0down = self._make_layer(block, input_channels, nb_filter[0])
        self.conv0_0up = self._make_layer(HierarchicalAttention, input_channels, nb_filter[0],pow(512, 2),512,num_blocks = 1)


        self.conv1_0 = self._make_layer(block, nb_filter[0],  nb_filter[1],0,0, num_blocks[1])
        #self.conv1_0down = self._make_layer(block, nb_filter[0],  nb_filter[1])
        self.conv1_0up = self._make_layer(HierarchicalAttention, nb_filter[0], nb_filter[1],pow(256, 2), 256,num_blocks = 1)

        self.conv2_0 = self._make_layer(block, nb_filter[1],  nb_filter[2],0,0, num_blocks[2])
       # self.conv2_0down = self._make_layer(block, nb_filter[1],  nb_filter[2])
        self.conv2_0up = self._make_layer(HierarchicalAttention, nb_filter[1],  nb_filter[2],pow(128, 2), 128,num_blocks = 1)



        self.conv3_0 = self._make_layer(block, nb_filter[2],  nb_filter[3],0,0, num_blocks[3])
       #self.conv3_0down = self._make_layer(block, nb_filter[2],  nb_filter[3])
        self.conv3_0up = self._make_layer(HierarchicalAttention, nb_filter[2],  nb_filter[3],pow(64 ,2), 64,num_blocks = 1 )

       
        self.convmid = middelmode(nb_filter[0])

        #self.conv4_0 = self._make_layer(block, nb_filter[3],  nb_filter[4])

        self.conv0_1 = self._make_layer(block, nb_filter[0] + nb_filter[1],  nb_filter[0],0,0,num_blocks[0])
        self.conv1_1 = self._make_layer(block, nb_filter[1] + nb_filter[2] + nb_filter[0],  nb_filter[1],0,0, num_blocks[0])
        self.conv2_1 = self._make_layer(block, nb_filter[2] + nb_filter[3] + nb_filter[2],  nb_filter[2],0,0, num_blocks[1])
        #self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4] + nb_filter[2],  nb_filter[3], num_blocks[2])

        self.conv0_2 = self._make_layer(block, nb_filter[0]*3, nb_filter[0],0,0,num_blocks[0])
        self.conv1_2 = self._make_layer(block, nb_filter[1]*2 + nb_filter[0] + nb_filter[2], nb_filter[1],0,0, num_blocks[0])
        #self.conv2_2 = self._make_layer(block, nb_filter[2]*2 + nb_filter[3]+ nb_filter[1], nb_filter[2], num_blocks[1])

        self.conv0_3 = self._make_layer(block, nb_filter[0]*3 + nb_filter[1], nb_filter[0],0,0,num_blocks[0])
       # self.conv1_3 = self._make_layer(block, nb_filter[1]*3 + nb_filter[2]+ nb_filter[0], nb_filter[1], num_blocks[0])

       # self.conv0_4 = self._make_layer(block, nb_filter[0]*4 + nb_filter[1], nb_filter[0])

        '''self.conv0_3_final = self._make_layer(block, nb_filter[0]*4, nb_filter[0],0,0,num_blocks[0])'''

        self.conv3_0_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        self.conv2_1_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        self.conv1_2_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)
        #self.conv0_3_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

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
 
          # data0-128  data-256 data-512
        
        
        x0_0 = self.conv0_0(data)           #x0_0256
       #x0_0down = self.conv0_0down(data0)   #x0_0down 128
        x0_0up = self.conv0_0up(data2)     #x0_0up 512

        x0_0M = self.mc0_0(torch.cat([x0_0,self.pool(x0_0up)],1),labels)  #mc

        x1_0 = self.conv1_0(self.pool(x0_0))   #x1_0128
       # x1_0down = self.conv1_0down(self.pool(x0_0down))   #x1_0down64
        x1_0up = self.conv1_0up(self.pool(x0_0up)) #x1_0up256

        x1_0M = self.mc1_0(torch.cat([x1_0,self.pool(x1_0up)],1),labels)   #mc

        x2_0 = self.conv2_0(self.pool(x1_0))     #x2_064
       # x2_0down = self.conv2_0down(self.pool(x1_0down))      #x2_0down32
        x2_0up = self.conv2_0up(self.pool(x1_0up))           #x2_0up128

        x2_0M = self.mc2_0(torch.cat([x2_0,self.pool(x2_0up)],1),labels)  #mc

        x3_0 = self.conv3_0(self.pool(x2_0))        #x3_0 32
       # x3_0down = self.conv3_0down(self.pool(x2_0down))        #x3_0down 16
        x3_0up = self.conv3_0up(self.pool(x2_0up))      #x3_0up 64

        x3_0M = self.mc3_0(torch.cat([x3_0,self.pool(x3_0up)],1),labels)

       
       
        
       
        x0_1 = self.conv0_1(torch.cat([x0_0M, self.up(x1_0M)], 1)) ################################

        #x1_1 = self.conv1_1(torch.cat([x1_0M, self.up(x2_0M),self.down(x0_1)], 1))     #######################################

        x1_1h,x1_1m,x1_1l = self.convmid(x0_1,x1_0M,x2_0M)   #x00m16,x10m32,x20m64

        x0_2 = self.conv0_2(torch.cat([x0_0M, x0_1, x1_1h], 1))  #########################################

       #x3_0M = self.mc3_0(x3_0,labels)   #mc

        x2_1 = self.conv2_1(torch.cat([x2_0M, self.up(x3_0M),x1_1l], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0M, x1_1m, self.pool(x0_2), self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0M, x0_2, x0_1, self.up(x1_2)], 1))

        #x4_0 = self.conv4_0(self.pool(x3_0))
        #x4_01 = self.DSC(x4_0)  #引入DSC模块

        #x3_1 = self.conv3_1(torch.cat([x3_0M, self.up(x4_0),self.down(x2_1)], 1))
        #x2_2 = self.conv2_2(torch.cat([x2_0M, x2_1, self.up(x3_1),self.down(x1_2)], 1))
        #x1_3 = self.conv1_3(torch.cat([x1_0M, x1_1, x1_2, self.up(x2_2),self.down(x0_3)], 1))
        #x0_4 = self.conv0_4(torch.cat([x0_0M, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

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


############################################################################################################################################################
# 假设您已经定义了DNANet类，以下是计算参数量的函数
'''def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 实例化模型
num_classes = 1  # 假设有10个分类
input_channels = 3  # 输入通道数，例如RGB图像
nb_filter = [16, 32, 64, 128, 256]  # 各个卷积层的过滤器数量
num_blocks = [2, 2, 2, 2]  # 每个层中的块的数量
model = DNANet(num_classes=num_classes, input_channels=input_channels, block=Res_CBAM_block, num_blocks=num_blocks, nb_filter=nb_filter)

# 计算参数量
num_parameters = count_parameters(model)
print(f'Number of parameters in the model: {num_parameters}')'''
###########################################################################################################################################################

'''import torch
import time

# 设置设备为 GPU (如果可用) 或 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化网络
model = DNANet(num_classes=1, input_channels=3, block=Res_CBAM_block, num_blocks=[2, 2, 2, 2], nb_filter=[4, 8, 16, 32, 64]).to(device)
model.eval()  # 切换到评估模式

# 随机生成输入数据 (根据网络的输入尺寸)
batch_size = 8  # 可以根据需要调整
input_size = (256, 256)  # 假设输入图片大小为 256x256

# 根据不同分辨率生成输入数据
data0 = torch.randn(batch_size, 3, input_size[0] // 2, input_size[1] // 2).to(device)  # 下采样输入
data = torch.randn(batch_size, 3, input_size[0], input_size[1]).to(device)  # 原始输入
data2 = torch.randn(batch_size, 3, input_size[0] * 2, input_size[1] * 2).to(device)  # 上采样输入
labels = torch.randint(0, 2, (batch_size, input_size[0], input_size[1])).to(device)  # 随机生成标签

# 增加一个通道维度以匹配模型的输入要求
labels = labels.unsqueeze(1)  # 变为 (batch_size, 1, height, width)

# 跑 100 次推理，测量时间
num_iterations = 300
start_time = time.time()

with torch.no_grad():  # 在推理时不计算梯度
    for _ in range(num_iterations):
        _ = model(data0, data, data2, labels)

end_time = time.time()

# 计算平均推理时间
total_time = end_time - start_time
avg_inference_time = total_time / num_iterations

# 计算 FPS
fps = num_iterations / total_time  # 计算 FPS
print(f"Average Inference Time: {avg_inference_time:.6f} seconds")
print(f"FPS: {fps:.2f}")'''

#######################################################################################################################################################################