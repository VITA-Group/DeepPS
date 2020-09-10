import torch

from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
import scipy.ndimage as pyimg
import os
import random
import math
import torch.nn.functional as F

from torch.utils import data
from torchvision import transforms as T

# discard partial lines by removing random patches from the full sketches
def random_discard(img):
    ht = img.size(2)
    min_ht = ht // 8
    max_ht = min_ht * 2
    for i in range(img.size(0)):
        for _ in range(np.random.randint(0, 3)):
            mask_w = np.random.randint(min_ht, max_ht)
            mask_h = np.random.randint(min_ht, max_ht)
            px = np.random.randint(0, ht-mask_w)
            py = np.random.randint(0, ht-mask_h)
            img[i,:,py:py+mask_h, px:px+mask_w] = 1.0
    return img

# We use PostprocessHED.m provided in pix2pix github page to simplify HED edges.
# The obtained binary lines with width of 1 are further smoothed by myDilateBlur to simulate the real sketches. 
# myDilateBlur()  for 256*256 and 64*64
# myDilateBlur(kernel_size=5, sigma=0.55) for 128*128
class MyDilateBlur(nn.Module):
    def __init__(self, kernel_size=7, channels=3, sigma=0.8):
        super(MyDilateBlur, self).__init__()
        self.kernel_size=kernel_size
        self.channels = channels
        # Set these to whatever you want for your gaussian filter
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_cord = torch.arange(self.kernel_size+0.)
        x_grid = x_cord.repeat(self.kernel_size).view(self.kernel_size, self.kernel_size)
        y_grid = x_grid.t()
        self.xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        self.mean = (self.kernel_size - 1)//2
        self.diff = -torch.sum((self.xy_grid - self.mean)**2., dim=-1)
        self.gaussian_filter = nn.Conv2d(in_channels=self.channels, out_channels=self.channels,
                                    kernel_size=self.kernel_size, groups=self.channels, bias=False)

        self.gaussian_filter.weight.requires_grad = False
        variance = sigma**2.
        gaussian_kernel = (1./(2.*math.pi*variance)) * torch.exp(self.diff /(2*variance))
        # note that normal gaussain_kernel use gaussian_kernel / torch.sum(gaussian_kernel)
        # here we multiply with 2 to make a small dilation
        gaussian_kernel = 2 * gaussian_kernel / torch.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel.view(1, 1, self.kernel_size, self.kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(self.channels, 1, 1, 1)
        self.gaussian_filter.weight.data = gaussian_kernel
        
    def forward(self, x):        
        y = self.gaussian_filter(F.pad(1-x, (self.mean,self.mean,self.mean,self.mean), "replicate")) 
        return 1 - 2 * torch.clamp(y, min=0, max=1)
    

# We implement dilation operations as convolutional layers with all-ones kernels of different radii r=(kernel_size-1)/2, followed by data clipping into [0,1] (cliping is in ConditionalDilate)
class OneDilate(nn.Module):
    def __init__(self, kernel_size=10, channels=3, gpu=True):
        super(OneDilate, self).__init__()
        self.kernel_size=kernel_size
        self.channels = channels
        gaussian_kernel = torch.ones(1, 1, self.kernel_size, self.kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(self.channels, 1, 1, 1)
        self.mean = (self.kernel_size - 1)//2
        self.gaussian_filter = nn.Conv2d(in_channels=self.channels, out_channels=self.channels,
                                    kernel_size=self.kernel_size, groups=self.channels, bias=False)
        if gpu:
            gaussian_kernel = gaussian_kernel.cuda()
        self.gaussian_filter.weight.data = gaussian_kernel
        self.gaussian_filter.weight.requires_grad = False
        
    def forward(self, x):
        x = F.pad((1-x)*0.5, (self.mean,self.mean,self.mean,self.mean), "replicate")
        return self.gaussian_filter(x)      

# Dilation results using the fractional radii are obtained by interpolating the results using the integer radii.
class ConditionalDilate(nn.Module):
    def __init__(self, max_kernel_size=21, channels=3, gpu=True):
        super(ConditionalDilate, self).__init__()
        
        self.max_kernel_size = max_kernel_size//2*2+1
        self.netBs = [OneDilate(i, gpu=gpu) for i in range(1,self.max_kernel_size+1,2)]
        
    def forward(self, x, l):
        l = min(self.max_kernel_size, max(1, l))
        lf = int(np.floor(l))
        if l == lf and l%2 == 1:
            out = self.netBs[(lf-1)//2](x)
        else:
            lf = lf - (lf+1)%2
            lc = lf + 2
            x1 = self.netBs[(lf-1)//2](x)
            x2 = self.netBs[(lc-1)//2](x)
            out = (x1 * (lc-l) + x2 * (l-lf))/2.0
        return 1 - 2 * torch.clamp(out, min=0, max=1)   
    
# random deformation    
def make_Gaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    
    temp = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    
    return temp/sum(sum(temp))

class RandomDeform(nn.Module):
    def __init__(self, input_size, grid_size, fiter_size, gpu=True):
        super(RandomDeform, self).__init__()
        
        self.grid_size = grid_size
        self.fiter_size = fiter_size
        self.input_size = input_size
        self.pad = nn.ReplicationPad2d(self.fiter_size)
        self.gpu = gpu
        
        gaussian_weights = torch.FloatTensor(make_Gaussian(2*self.fiter_size+1, fwhm = self.fiter_size))
        self.filter = nn.Conv2d(1, 1, kernel_size=(2*self.fiter_size+1,2*self.fiter_size+1),bias=False)
        self.filter.weight[0].data[:,:,:] = gaussian_weights

        self.P_basis = torch.zeros(2,self.input_size, self.input_size)
        for k in range(2):
            for i in range(self.input_size):
                for j in range(self.input_size):
                    self.P_basis[k,i,j] = k*i/(self.input_size-1.0)+(1.0-k)*j/(self.input_size-1.0)
        
    def create_grid(self, x, max_move):
        max_offset = 2.0*max_move/self.input_size
        P = torch.autograd.Variable(torch.zeros(1,2,self.input_size, self.input_size),requires_grad=False)
        P = P.cuda() if self.gpu else P
        P[0,:,:,:] = self.P_basis*2-1
        P = P.expand(x.size(0),2,self.input_size, self.input_size)
        offset_x = torch.autograd.Variable(torch.randn(x.size(0),1,self.grid_size, 
                                                       self.grid_size))
        offset_y = torch.autograd.Variable(torch.randn(x.size(0),1,self.grid_size, 
                                                       self.grid_size))
        offset_x = offset_x.cuda() if self.gpu else offset_x
        offset_y = offset_y.cuda() if self.gpu else offset_y
        offset_x_filter = self.filter(self.pad(offset_x)) * max_offset
        offset_y_filter = self.filter(self.pad(offset_y)) * max_offset
        offset_x_filter = torch.clamp(offset_x_filter,min=-max_offset,max=max_offset)
        offset_y_filter = torch.clamp(offset_y_filter,min=-max_offset,max=max_offset)

        grid = torch.cat((offset_x_filter,offset_y_filter), 1)
        grid = F.interpolate(grid, [self.input_size,self.input_size], mode='bilinear')
        
        grid = torch.clamp(grid + P, min=-1, max=1)
        grid = torch.transpose(grid,1,2)
        grid = torch.transpose(grid,2,3)
        return grid

    def forward(self, x, max_move):
        grid = self.create_grid(x, max_move)
        x_sampled = F.grid_sample(x, grid)
        return x_sampled
