from __future__ import print_function
import numpy as np
import random
import cv2
import math
import os

import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm2d, LeakyReLU, ConvTranspose2d, ReLU, Tanh, InstanceNorm2d
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import ReflectionPad2d, ReplicationPad2d
from torch.nn.utils import spectral_norm

from utils import to_var, to_data, weights_init, visualize, load_image, save_image, binarize, get_mask
from roughSketchSyn import ConditionalDilate, MyDilateBlur, random_discard, RandomDeform

id = 0 # for saving network output to file during training

class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, style_dim, in_channel):
        super(AdaptiveInstanceNorm, self).__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = nn.Linear(style_dim, in_channel * 2)

        self.style.bias.data[:in_channel] = 1
        self.style.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out

class StyleGenerator(nn.Module):
    def __init__(self, code_dim=128, n_mlp=4):
        super(StyleGenerator, self).__init__()

        layers = []
        layers.append(Linear(1, code_dim))
        layers.append(LeakyReLU(0.2))
        for i in range(n_mlp-1):
            layers.append(nn.Linear(code_dim, code_dim))
            layers.append(LeakyReLU(0.2))

        self.style = nn.Sequential(*layers)

    def forward(self, l):
        return self.style(l)
    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

class UpsampleConLayer(nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = ReflectionPad2d(reflection_padding)
        self.conv2d = Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out 
    
class RCBBlock(nn.Module):
    def __init__(self, in_channels, out_channels=64, kernel_size=3, padding=1, embeding_size=128):
        super(RCBBlock, self).__init__()
        
        self.relu = LeakyReLU(0.2)
        self.pad = ReflectionPad2d(padding=padding)
        self.conv = Conv2d(out_channels=out_channels, kernel_size=kernel_size, stride=2,
                              padding=0, in_channels=in_channels)

        self.bn = AdaptiveInstanceNorm(embeding_size, out_channels)
        
    def forward(self, x, y):
        return self.bn(self.conv(self.pad(self.relu(x))), y)
    
class RDCBBlock(nn.Module):
    def __init__(self, in_channels, out_channels=64, kernel_size=3, padding=1, embeding_size=128, up=False):
        super(RDCBBlock, self).__init__()
        
        self.relu = ReLU()
        if up == False:
            self.dconv = ConvTranspose2d(out_channels=out_channels, kernel_size=kernel_size, 
                                     stride=2, padding=padding, in_channels=in_channels) 
        else:
            self.dconv = UpsampleConLayer(out_channels=out_channels, kernel_size=kernel_size, 
                                     stride=1, in_channels=in_channels, upsample=2)
        
        self.bn = AdaptiveInstanceNorm(embeding_size, out_channels)
            
    def forward(self, x, y):
        return self.bn(self.dconv(self.relu(x)), y)

class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, weight=0.1, is_bias=True, embeding_size=128):
        super(ResnetBlock, self).__init__()
        # Attributes
        self.is_bias = is_bias
        self.weight = weight
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        self.actvn = LeakyReLU(0.2)
        self.embeding_size = embeding_size
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        self.bn1 = AdaptiveInstanceNorm(self.embeding_size, self.fhidden)
        self.bn2 = AdaptiveInstanceNorm(self.embeding_size, self.fout)
            
        self.pad_0 = ReflectionPad2d(padding=1)
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=0)
        self.pad_1 = ReflectionPad2d(padding=1)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=0, bias=is_bias)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)

    def forward(self, x, y):
        x_s = self._shortcut(x, y)
        dx = self.bn1(self.conv_0(self.pad_0(self.actvn(x))), y)
        dx = self.bn2(self.conv_1(self.pad_1(self.actvn(dx))), y)
        out = x_s + self.weight*dx

        return out

    def _shortcut(self, x, y):
        if self.learned_shortcut:
            x_s = self.bn2(self.conv_s(x), y)
        else:
            x_s = x
        return x_s    
    
class Generator64(nn.Module):
    def __init__(self, nef = 64, n_layers = 3, in_channels=3, oneconv = True, grayedge = True, 
                 embeding_size=128, n_mlp=4):
        super(Generator64, self).__init__()
        
        self.oneconv = oneconv
        self.grayedge = grayedge
        self.rgbchannels = 3
        self.edgechannels = 3
        if self.grayedge:
            self.edgechannels = 1
        if self.oneconv:
            self.edgechannels = self.edgechannels + self.rgbchannels
        self.embeding_size = embeding_size
        self.embedding = StyleGenerator(embeding_size, n_mlp)
        
        modelList = []
        # 3*64*64 x --> 64*64*64 x1
        self.pad1 = ReflectionPad2d(padding=1)
        self.conv1 = Conv2d(out_channels=nef, kernel_size=3, padding=0, in_channels=in_channels)
        # 64*64*64 x1 --> 128*32*32 x2
        self.rcb1 = RCBBlock(nef, nef*2, 3, 1, embeding_size) 
        # 128*32*32 x2 --> 256*16*16 x3
        self.rcb2 = RCBBlock(nef*2, nef*4, 3, 1, embeding_size)        
        # 256*16*16 x3 --> 512*8*8 x4
        self.rcb3 = RCBBlock(nef*4, nef*8, 3, 1, embeding_size)
        # 512*8*8 x4 --> 512*8*8 x5
        for n in range(n_layers):
            modelList.append(ResnetBlock(nef*8, nef*8, weight=1.0, embeding_size=embeding_size))  
        # 512*8*8 x5 --> 256*16*16
        self.rdcb3 = RDCBBlock(nef*8, nef*4, 3, 1, embeding_size, True)    
        # 512*16*16 x6 --> 128*32*32 
        self.rdcb2 = RDCBBlock(nef*8, nef*2, 3, 1, embeding_size, True)
        # 256*32*32 x7 --> 64*64*64 
        self.rdcb1 = RDCBBlock(nef*4, nef, 3, 1, embeding_size, True)

        self.resblocks = nn.Sequential(*modelList)
        # 64*64*64 x6 --> 6*64*64 
        
        self.pad2 = ReflectionPad2d(padding=1)
        self.relu = ReLU()
        self.conv2 = Conv2d(out_channels=self.edgechannels, kernel_size=3, padding=0, in_channels=nef*2)
        self.tanh = Tanh()     
        self.conv3 = Conv2d(out_channels=self.rgbchannels, kernel_size=3, padding=0, in_channels=nef*2)
        
    def forward_feature(self, x, l, f=None):
        y = self.embedding(l)
        x1 = self.conv1(self.pad1(x))
        x2 = self.rcb1(x1, y)
        x3 = self.rcb2(x2, y)
        x4 = self.rcb3(x3, y)
        for resblock in self.resblocks:
            x4 = resblock(x4, y)
        x6 = torch.cat((self.rdcb3(x4, y), x3), dim=1)
        x7 = torch.cat((self.rdcb2(x6, y), x2), dim=1)
        return torch.cat((self.rdcb1(x7, y), x1), dim=1)
    
    def forward(self, x, l, f=None):
        y = self.embedding(l)
        x1 = self.conv1(self.pad1(x))
        x2 = self.rcb1(x1, y)
        x3 = self.rcb2(x2, y)
        x4 = self.rcb3(x3, y)
        for resblock in self.resblocks:
            x4 = resblock(x4, y)        
        x6 = torch.cat((self.rdcb3(x4, y), x3), dim=1)
        x7 = torch.cat((self.rdcb2(x6, y), x2), dim=1)
        x8 = self.pad2(torch.cat((self.rdcb1(x7, y), x1), dim=1))
        edge = self.tanh(self.conv2(self.relu(x8)))
        # oneconv: use one conv layer to compute both S_gen and I_gen
        # grayedge: make sure S_gen is black and white image 
        if self.oneconv:
            if self.grayedge:
                edge = torch.cat((edge[:,0:1], edge[:,0:1], edge), dim=1)       
            return edge
        else:
            rgb = self.tanh(self.conv3(self.relu(x8)))
            if self.grayedge:
                edge = edge.expand(1, 3, 1, 1)
            return torch.cat((edge, rgb), dim=1)

class Generator128(nn.Module):
    def __init__(self, nef = 64, n_layers = 3, in_channels=3, oneconv = True, grayedge = True, 
                 embeding_size=128, n_mlp=4):
        super(Generator128, self).__init__()

        self.oneconv = oneconv
        self.grayedge = grayedge
        self.rgbchannels = 3
        self.edgechannels = 3
        if self.grayedge:
            self.edgechannels = 1
        if self.oneconv:
            self.edgechannels = self.edgechannels + self.rgbchannels   
        self.embeding_size = embeding_size
        self.embedding = StyleGenerator(embeding_size, n_mlp)
        
        modelList = []
        # 3*128*128 x --> 64*128*128 x1
        self.pad1 = ReflectionPad2d(padding=1)
        self.conv1 = Conv2d(out_channels=nef, kernel_size=3, padding=0, in_channels=in_channels)
        # 64*128*128 x1 --> 128*64*64 x2
        self.rcb1 = RCBBlock(nef, nef*2, 3, 1, embeding_size)
        # 128*64*64 x2+y --> 128*64*64 x3
        for n in range(n_layers):
            modelList.append(ResnetBlock(nef*2, nef*2, weight=1.0, embeding_size=embeding_size))
        # 128*64*64 x3 --> 64*128*128 
        self.rdcb1 = RDCBBlock(nef*2, nef, 3, 1, embeding_size, True)

        self.resblocks = nn.Sequential(*modelList)
        # 128*128*128 x4 --> 6*128*128 
        self.pad2 = ReflectionPad2d(padding=1)
        self.relu = ReLU()
        self.conv2 = Conv2d(out_channels=self.edgechannels, kernel_size=3, padding=0, in_channels=nef*2)
        self.tanh = Tanh()     
        self.conv3 = Conv2d(out_channels=self.rgbchannels, kernel_size=3, padding=0, in_channels=nef*2)
        
    def forward_feature(self, x, l, f):
        y = self.embedding(l)
        x1 = self.conv1(self.pad1(x))
        x2 = self.rcb1(x1, y)
        x3 = x2 + f
        for resblock in self.resblocks:
            x3 = resblock(x3, y)
        return torch.cat((self.rdcb1(x3, y), x1), dim=1)
        
    def forward(self, x, l, f):
        y = self.embedding(l)
        x1 = self.conv1(self.pad1(x))
        x2 = self.rcb1(x1, y)
        x3 = x2 + f
        for resblock in self.resblocks:
            x3 = resblock(x3, y)
        x4 = self.pad2(torch.cat((self.rdcb1(x3, y), x1), dim=1))
        edge = self.tanh(self.conv2(self.relu(x4)))
        # oneconv: use one conv layer to compute both S_gen and I_gen
        # grayedge: make sure S_gen is black and white image 
        if self.oneconv:
            if self.grayedge:
                edge = torch.cat((edge[:,0:1], edge[:,0:1], edge), dim=1)      
            return edge
        else:
            rgb = self.tanh(self.conv3(self.relu(x4)))
            if self.grayedge:
                edge = edge.expand(1, 3, 1, 1)
            return torch.cat((edge, rgb), dim=1)

class Generator256(nn.Module):
    def __init__(self, nef=64, n_layers = 3, in_channels=3, oneconv = True, grayedge = True, 
                 embeding_size=128, n_mlp=4):
        super(Generator256, self).__init__()

        self.oneconv = oneconv
        self.grayedge = grayedge
        self.rgbchannels = 3
        self.edgechannels = 3
        if self.grayedge:
            self.edgechannels = 1
        if self.oneconv:
            self.edgechannels = self.edgechannels + self.rgbchannels  
        self.embeding_size = embeding_size
        self.embedding = StyleGenerator(embeding_size, n_mlp)
        
        modelList = []
        # 3*256*256 x --> 64*256*256 x1
        self.pad1 = ReflectionPad2d(padding=1)
        self.conv1 = Conv2d(out_channels=nef, kernel_size=3, padding=0, in_channels=in_channels)
        # 64*256*256 x1 --> 128*128*128 x2
        self.rcb1 = RCBBlock(nef, nef*2, 3, 1, embeding_size) 
        # 128*128*128 x2+y --> 128*128*128 x3
        for n in range(n_layers):
            modelList.append(ResnetBlock(nef*2, nef*2, weight=1.0, embeding_size=embeding_size))  
        # 128*128*128 x3 --> 64*256*256 
        self.rdcb1 = RDCBBlock(nef*2, nef, 3, 1, embeding_size, True)
        self.resblocks = nn.Sequential(*modelList)
        # 64*256*256 x4 --> 6*256*256
        self.pad2 = ReflectionPad2d(padding=1)
        self.relu = ReLU()
        self.conv2 = Conv2d(out_channels=self.edgechannels, kernel_size=3, padding=0, in_channels=nef*2)
        self.tanh = Tanh()     
        self.conv3 = Conv2d(out_channels=self.rgbchannels, kernel_size=3, padding=0, in_channels=nef*2)
    
    def forward(self, x, l, f):
        y = self.embedding(l)
        x1 = self.conv1(self.pad1(x))
        x2 = self.rcb1(x1, y)
        x3 = x2 + f
        for resblock in self.resblocks:
            x3 = resblock(x3, y) 
        x4 = self.pad2(torch.cat((self.rdcb1(x3, y), x1), dim=1))
        edge = self.tanh(self.conv2(self.relu(x4)))
        # oneconv: use one conv layer to compute both S_gen and I_gen
        # grayedge: make sure S_gen is black and white image 
        if self.oneconv:
            if self.grayedge:
                edge = torch.cat((edge[:,0:1], edge[:,0:1], edge), dim=1)     
            return edge
        else:
            rgb = self.tanh(self.conv3(self.relu(x4)))
            if self.grayedge:
                edge = edge.expand(1, 3, 1, 1)
            return torch.cat((edge, rgb), dim=1)  

class DiscriminatorSN(nn.Module):
    def __init__(self, in_channels, out_channels, ndf=64, n_layers=3, input_size=64, useFC=False):
        super(DiscriminatorSN, self).__init__()
        
        modelList = []       
        kernel_size = 4
        padding = int(np.ceil((kernel_size - 1)/2))
        modelList.append(ReflectionPad2d(padding=padding))
        modelList.append(spectral_norm(Conv2d(out_channels=ndf, kernel_size=kernel_size, stride=2,
                              padding=0, in_channels=in_channels)))
        modelList.append(LeakyReLU(0.2))
        self.useFC = useFC
        
        # 32*32 --> 16*16 --> 8*8
        size = input_size//2
        nf_mult = 1
        for n in range(1, n_layers):
            size = size // 2
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            modelList.append(ReflectionPad2d(padding=padding))
            modelList.append(spectral_norm(Conv2d(out_channels=ndf * nf_mult, kernel_size=kernel_size, stride=2,
                                  padding=0, in_channels=ndf * nf_mult_prev)))
            modelList.append(LeakyReLU(0.2))
        # 7*7
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        modelList.append(ReflectionPad2d(padding=padding))
        modelList.append(spectral_norm(Conv2d(out_channels=ndf * nf_mult, kernel_size=kernel_size, stride=1,
                              padding=0, in_channels=ndf * nf_mult_prev)))
        # 6*6
        modelList.append(LeakyReLU(0.2))
        modelList.append(ReflectionPad2d(padding=padding))
        modelList.append(spectral_norm(Conv2d(out_channels=out_channels, kernel_size=kernel_size, stride=1,
                              padding=0, in_channels=ndf * nf_mult)))

        self.model = nn.Sequential(*modelList)
        self.fc = spectral_norm(nn.Linear((size-2)*(size-2)*out_channels, 1))
        
    def forward(self, x):
        out = self.model(x).view(x.size(0), -1)
        if self.useFC:
            out = self.fc(out)
        return out.view(-1)              
        
# editing: G_channel = 7, D_channel = 7
# synthesis: G_channel = 3, D_channel = 6
class PSGAN(nn.Module):
    def __init__(self, G_channels =7, G_nlayers = 3, G_nf = 64, 
                 D_channels =7, D_nf = 64, D_nlayers = 3, 
                 max_dilate = 21, max_level = 3, img_size =256, gpu=True):
        super(PSGAN, self).__init__()
        
        self.G_channels = G_channels
        self.G_nlayers = G_nlayers
        self.G_nf = G_nf
        self.D_nlayers = D_nlayers
        self.D_channels = D_channels
        self.D_nf = D_nf
        self.max_dilate = max_dilate
        self.gpu = gpu
        self.weight_rec = 100.0
        self.weight_adv = 1.0
        self.weight_perc = [1,0.5]
        self.hinge = 10.0
        self.max_level = max_level
        self.img_size = img_size # (max_level,img_size): (1,64), (2,128), (3,256)
        self.L1loss = nn.L1Loss()
        self.L2loss = nn.MSELoss()

        self.G64 = Generator64(nef=self.G_nf, n_layers = self.G_nlayers, in_channels=self.G_channels)
        self.G128 = Generator128(nef=self.G_nf, n_layers = self.G_nlayers, in_channels=self.G_channels)
        self.G256 = Generator256(nef=self.G_nf, n_layers = self.G_nlayers, in_channels=self.G_channels)
        
        self.D64 = DiscriminatorSN(in_channels=self.D_channels, out_channels=self.D_nf, ndf=self.D_nf, n_layers = self.D_nlayers, input_size=64)
        self.D128 = DiscriminatorSN(in_channels=self.D_channels, out_channels=self.D_nf, ndf=self.D_nf, n_layers = self.D_nlayers+1, input_size=128)
        self.D256 = DiscriminatorSN(in_channels=self.D_channels, out_channels=self.D_nf, ndf=self.D_nf, n_layers = self.D_nlayers+2, input_size=256)
        
        self.edgeSmooth1 = MyDilateBlur() # for 256 and 64
        self.edgeSmooth2 = MyDilateBlur(kernel_size=5, sigma=0.55) # for 128
        self.deform = RandomDeform(self.img_size, 16, 4, gpu=self.gpu)
        self.edgedilate = ConditionalDilate(self.max_dilate, gpu=self.gpu) 
        
        self.trainerG64 = torch.optim.Adam(self.G64.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.trainerG128 = torch.optim.Adam(self.G128.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.trainerG256 = torch.optim.Adam(self.G256.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.trainerD64 = torch.optim.Adam(self.D64.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.trainerD128 = torch.optim.Adam(self.D128.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.trainerD256 = torch.optim.Adam(self.D256.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
    # FOR TESTING
    def load_generator(self, filepath, filename): 
        device = None if self.gpu else torch.device('cpu') 
        self.G64.load_state_dict(torch.load(os.path.join(filepath, filename+'-G64.ckpt'), map_location=device))
        self.G128.load_state_dict(torch.load(os.path.join(filepath, filename+'-G128.ckpt'), map_location=device))
        self.G256.load_state_dict(torch.load(os.path.join(filepath, filename+'-G256.ckpt'), map_location=device))
    
    # for 256*256 images only
    # if input is 64*64, then the size of S256_in, I256_in and M will be 64*64
    def prepare_testing_images(self, S, I=None, M=None, label=1.0):
        S256 = self.edgeSmooth1(S)
        S256_in = self.edgedilate(S256, label)
        S128_in = F.interpolate(S256_in, mode='bilinear', scale_factor=0.5, align_corners=True) 
        S64_in = F.interpolate(S256_in, mode='bilinear', scale_factor=0.25, align_corners=True) 
        # for synthesis
        if I is None:
            return S256_in, S128_in, S64_in
        # for editing
        assert(M is not None)
        I128 = F.interpolate(I, mode='bilinear', scale_factor=0.5, align_corners=True) 
        I64 = F.interpolate(I, mode='bilinear', scale_factor=0.25, align_corners=True) 
        M128 = binarize(F.interpolate(M, mode='bilinear', scale_factor=0.5, 
                                      align_corners=True), threshold=-1)                     
        M64 = binarize(F.interpolate(M, mode='bilinear', scale_factor=0.25, 
                                     align_corners=True), threshold=-1) 
        I256_in = I * (1-M)/2
        S256_in = S256_in * (1+M)/2
        I128_in = I128 * (1-M128)/2
        S128_in = S128_in * (1+M128)/2
        I64_in = I64 * (1-M64)/2
        S64_in = S64_in * (1+M64)/2
        return [torch.cat((S256_in, I256_in, M), dim=1),
            torch.cat((S128_in, I128_in, M128), dim=1),
            torch.cat((S64_in, I64_in, M64), dim=1)]
    
    # for 256*256 images only
    # if want to test on 64*64, 'outpust = self.G64(input256, l).detach()'
    # and remove self.G128 and self.G256
    def forward_editing(self, S, I, M, l):
        label = (self.max_dilate-1.0) * l + 1.0
        l = torch.tensor(l).float().expand(1, 1)        
        l = l.cuda() if self.gpu else l
        input256, input128, input64 = self.prepare_testing_images(S, I, M, label)
        with torch.no_grad():
            feature64 = self.G64.forward_feature(input64, l).detach()
            feature128 = self.G128.forward_feature(input128, l, feature64).detach()
            outputs = self.G256(input256, l, feature128).detach()
            S_gen =  outputs[:,0:3] * (1+M)/2
            I_gen = outputs[:,3:6]
        return [S_gen, I_gen]
    
    # for 256*256 images only
    # if want to test on 64*64, 'outpust = self.G64(input256, l).detach()'
    # and remove self.G128 / self.G256
    def forward_synthesis(self, S, l):
        label = (self.max_dilate-1.0) * l + 1.0
        l = torch.tensor(l).float().expand(1, 1)
        l = l.cuda() if self.gpu else l
        input256, input128, input64 = self.prepare_testing_images(S, label=label)
        with torch.no_grad():
            feature64 = self.G64.forward_feature(input64, l).detach()
            feature128 = self.G128.forward_feature(input128, l, feature64).detach()
            outputs = self.G256(input256, l, feature128).detach()
            S_gen =  outputs[:,0:3]
            I_gen = outputs[:,3:6]
        return [S_gen, I_gen]
    
    # FOR TRAINING
    # init weight
    def init_networks(self, weights_init):
        self.G64.apply(weights_init)
        self.G128.apply(weights_init)
        self.G256.apply(weights_init)
        self.D64.apply(weights_init)
        self.D128.apply(weights_init)
        self.D256.apply(weights_init)

    def save_model(self, filepath, filename, level=[1,2,3]):
        if 1 in level:
            torch.save(self.G64.state_dict(), os.path.join(filepath, filename+'-G64.ckpt'))
            torch.save(self.D64.state_dict(), os.path.join(filepath, filename+'-D64.ckpt'))
        if 2 in level:
            torch.save(self.G128.state_dict(), os.path.join(filepath, filename+'-G128.ckpt'))
            torch.save(self.D128.state_dict(), os.path.join(filepath, filename+'-D128.ckpt'))
        if 3 in level:
            torch.save(self.G256.state_dict(), os.path.join(filepath, filename+'-G256.ckpt'))
            torch.save(self.D256.state_dict(), os.path.join(filepath, filename+'-D256.ckpt'))

    def get_network_by_level(self, level):
        if level == 1:
            return self.G64, self.D64, self.trainerG64, self.trainerD64
        elif level == 2:
            return self.G128, self.D128, self.trainerG128, self.trainerD128
        else:
            return self.G256, self.D256, self.trainerG256, self.trainerD256
    
    
    def prepare_training_images64(self, S, I, l, label, M=None):
        # ground truth fine sketches at resolution 64
        # edgeSmooth1 to make sure that each line is of about 3 pixel width
        S64 = self.edgeSmooth1(S)
        S64_in = self.edgedilate(random_discard(self.deform(S64, label-1)), label)
        # for synthesis
        if M is None:
            return S64_in, S64, I, None
        # for editing
        I64_in = I * (1-M)/2
        S64 = S64 * (1+M)/2 
        S64_in = S64_in * (1+M)/2 
        return S64_in, I64_in, S64, I, M, None
    
    def prepare_training_images128(self, S, I, l, label, level=2, M=None):
        # ground truth fine sketches at resolution 128 and 64
        # edgeSmooth to make sure that each line is of about 3 pixel width in each resolution
        S128 = self.edgeSmooth1(S)
        S64 = self.edgeSmooth2(F.interpolate(self.edgeSmooth2(S), mode='bilinear', scale_factor=0.5, align_corners=True))
        # synthesized rough sketches at resolution 128 and 64
        S128_in = self.edgedilate(random_discard(self.deform(S128, label-1)), label)
        S64_in = F.interpolate(S128_in, mode='bilinear', scale_factor=0.5, align_corners=True)
        I64 = F.interpolate(I, mode='bilinear', scale_factor=0.5, align_corners=True) 
        # for synthesis
        if M is None:
            if level == 1:
                return S64_in, S64, I64, None
            with torch.no_grad():
                feature64 = self.G64.forward_feature(S64_in, l).detach()
            return S128_in, S128, I, feature64
        
        # for editing
        M64 = binarize(F.interpolate(M, mode='bilinear', scale_factor=0.5, 
                                      align_corners=True), threshold=-1)
        I128_in = I * (1-M)/2
        S128 = S128 * (1+M)/2
        S128_in = S128_in * (1+M)/2
        I64_in = I64 * (1-M64)/2
        S64 = S64 * (1+M64)/2
        S64_in = S64_in * (1+M64)/2
        
        if level == 1:
            return S64_in, I64_in, S64, I64, M64, None
        with torch.no_grad():
            feature64 = self.G64.forward_feature(torch.cat((S64_in, I64_in, M64), dim=1), l).detach()
        return S128_in, I128_in, S128, I, M128, feature64
    
    def prepare_training_images256(self, S, I, l, label, level=3, M=None):
        # ground truth fine sketches at resolution 256, 128 and 64
        # edgeSmooth to make sure that each line is of about 3 pixel width in each resolution
        S256 = self.edgeSmooth1(S)
        S128 = self.edgeSmooth2(F.interpolate(self.edgeSmooth2(S), mode='bilinear', scale_factor=0.5, align_corners=True))
        S64 = self.edgeSmooth1(F.interpolate(self.edgeSmooth1(S), mode='bilinear', scale_factor=0.25, align_corners=True))
        # synthesized rough sketches at resolution 256, 128 and 64
        S256_in = self.edgedilate(random_discard(self.deform(S256, label-1)), label)
        S128_in = F.interpolate(S256_in, mode='bilinear', scale_factor=0.5, align_corners=True)
        S64_in = F.interpolate(S256_in, mode='bilinear', scale_factor=0.25, align_corners=True)
        I128 = F.interpolate(I, mode='bilinear', scale_factor=0.5, align_corners=True) 
        I64 = F.interpolate(I, mode='bilinear', scale_factor=0.25, align_corners=True) 
        # for synthesis
        if M is None:
            if level == 1:
                return S64_in, S64, I64, None
            with torch.no_grad():
                feature64 = self.G64.forward_feature(S64_in, l).detach()
            if level == 2:
                return S128_in, S128, I128, feature64
            with torch.no_grad():
                feature128 = self.G128.forward_feature(S128_in, l, feature64).detach()           
            return S256_in, S256, I, feature128
        # for editing
        M128 = binarize(F.interpolate(M, mode='bilinear', scale_factor=0.5, 
                                      align_corners=True), threshold=-1)                     
        M64 = binarize(F.interpolate(M, mode='bilinear', scale_factor=0.25, 
                                     align_corners=True), threshold=-1) 
        I256_in = I * (1-M)/2
        S256 = S256 * (1+M)/2
        S256_in = S256_in * (1+M)/2
        I128_in = I128 * (1-M128)/2
        S128 = S128 * (1+M128)/2
        S128_in = S128_in * (1+M128)/2
        I64_in = I64 * (1-M64)/2
        S64 = S64 * (1+M64)/2
        S64_in = S64_in * (1+M64)/2
        
        if level == 1:
            return S64_in, I64_in, S64, I64, M64, None
        with torch.no_grad():
            feature64 = self.G64.forward_feature(torch.cat((S64_in, I64_in, M64), dim=1), l).detach()
        if level == 2:
            return S128_in, I128_in, S128, I128, M128, feature64
        with torch.no_grad():
            feature128 = self.G128.forward_feature(torch.cat((S128_in, I128_in, M128), dim=1), l, feature64).detach()
        return S256_in, I256_in, S256, I, M, feature128

    # prepare training images at specified level
    def prepare_training_images(self, S, I, l, label, level=3, M=None):
        if self.max_level == 1:
            return self.prepare_training_images64(S, I, l, label, M)
        elif self.max_level == 2:
            return self.prepare_training_images128(S, I, l, label, level, M)
        else:
            return self.prepare_training_images256(S, I, l, label, level, M)
    
    def update_synthesis_discriminator(self, S_in, S_gt, I_gt, l, level, feature=None):
        G, D, _, trainerD = self.get_network_by_level(level)
        with torch.no_grad():
            real_concat = torch.cat((S_gt, I_gt), dim=1)
            fake_concat = G(S_in, l, feature)
        real_output = D(real_concat)
        fake_output = D(fake_concat)
        L_D = self.weight_adv*((F.relu(self.hinge-real_output)).mean() + 
                               (F.relu(self.hinge+fake_output)).mean())
        trainerD.zero_grad()
        L_D.backward()
        trainerD.step()
        return L_D.data.mean()    
    
    def update_editing_discriminator(self, S_in, I_in, S_gt, I_gt, M, l, level, feature=None):
        G, D, _, trainerD = self.get_network_by_level(level)
        with torch.no_grad():
            real_gen = torch.cat((S_gt, I_gt), dim=1)
            real_concat = torch.cat((real_gen, M), dim=1)
            fake_gen = G(torch.cat((S_in, I_in, M), dim=1), l, feature)
            fake_gen[:,0:3] = fake_gen[:,0:3] * (1+M)/2
            fake_concat = torch.cat((fake_gen, M), dim=1)  
        real_output = D(real_concat)
        fake_output = D(fake_concat)
        L_D = self.weight_adv*((F.relu(self.hinge-real_output)).mean() + 
                               (F.relu(self.hinge+fake_output)).mean())
        trainerD.zero_grad()
        L_D.backward()
        trainerD.step()
        return L_D.data.mean()       

    def update_synthesis_generator(self, S_in, S_gt, I_gt, l, level, VGGfeatures, feature=None, netF=None):
        G, D, trainerG, _ = self.get_network_by_level(level)
        with torch.no_grad():
            real_Phi = VGGfeatures(I_gt)
            real_concat = torch.cat((S_gt, I_gt), dim=1)
        fake_concat = G(S_in, l, feature)
        fake_output = D(fake_concat)
        L_Gadv = -self.weight_adv*fake_output.mean()        
        fake_Phi = VGGfeatures(fake_concat[:,3:6])
        L_perc = sum([self.weight_perc[i] * self.L2loss(A, real_Phi[i]) for i,A in enumerate(fake_Phi)])
        L_rec = self.weight_rec * self.L1loss(fake_concat, real_concat)
        if netF is not None:
            I_out = netF(fake_concat[:,0:3])
            fake_Phi2 = VGGfeatures(I_out)
            L_perc = L_perc + sum([self.weight_perc[i] * self.L2loss(A, real_Phi[i]) for i,A in enumerate(fake_Phi2)])
            L_rec = L_rec + self.weight_rec * self.L1loss(I_out, I_gt)
        L_G = L_Gadv + L_perc + L_rec
        
        trainerG.zero_grad()
        L_G.backward()
        trainerG.step()
        
        ## for debug
        #global id      
        #if id % 700 == 0:
        #    viz_img = to_data(torch.cat((I_gt[0], fake_concat[0,0:3], fake_concat[0,3:6]), dim=2))
        #    save_image(viz_img, '../output/synthesized%d.jpg'%id)
        #id += 1
        
        return L_Gadv.data.mean(), L_perc.data.mean(), L_rec.data.mean()     

    def update_editing_generator(self, S_in, I_in, S_gt, I_gt, M, l, level, VGGfeatures, feature=None, netF=None):
        G, D, trainerG, _ = self.get_network_by_level(level)
        with torch.no_grad():
            real_Phi = VGGfeatures(I_gt)
            real_gen = torch.cat((S_gt, I_gt), dim=1)
        fake_gen = G(torch.cat((S_in, I_in, M), dim=1), l, feature)
        fake_gen[:,0:3] = fake_gen[:,0:3] * (1+M)/2
        fake_concat = torch.cat((fake_gen, M), dim=1)  
        fake_output = D(fake_concat)
        L_Gadv = -self.weight_adv*fake_output.mean()   
        fake_Phi = VGGfeatures(fake_gen[:,3:6])
        L_perc = sum([self.weight_perc[i] * self.L2loss(A, real_Phi[i]) for i,A in enumerate(fake_Phi)])
        L_rec = self.weight_rec * self.L1loss(fake_gen, real_gen)
        if netF is not None:
            I_out = netF(torch.cat((fake_gen[:,0:3], I_in, M), dim=1))
            fake_Phi2 = VGGfeatures(I_out)
            L_perc = L_perc + sum([self.weight_perc[i] * self.L2loss(A, real_Phi[i]) for i,A in enumerate(fake_Phi2)])
            L_rec = L_rec + self.weight_rec * self.L1loss(I_out, I_gt)
        L_G = L_Gadv + L_perc + L_rec
        
        trainerG.zero_grad()
        L_G.backward()
        trainerG.step()
        
        ## for debug
        #global id
        #if id % 700 == 0:
        #    viz_img = to_data(torch.cat((I_gt[0], fake_gen[0,0:3], fake_gen[0,3:6]), dim=2))
        #    save_image(viz_img, '../output/edited%d.jpg'%id)
        #id += 1
        
        return L_Gadv.data.mean(), L_perc.data.mean(), L_rec.data.mean()
    
    def synthesis_one_pass(self, S, I, l, level, VGGfeatures, netF=None):
        label = (self.max_dilate-1.0) * l + 1.0
        l = torch.tensor(l).float().expand(S.size(0), 1)
        l = l.cuda() if self.gpu else l
        S_in, S_gt, I_gt, feature = self.prepare_training_images(S, I, l, label, level)
        L_D = self.update_synthesis_discriminator(S_in, S_gt, I_gt, l, level, feature)
        L_Gadv, L_perc, L_rec = self.update_synthesis_generator(S_in, S_gt, I_gt, l, level, VGGfeatures, feature, netF)
        return [L_D, L_Gadv, L_perc, L_rec]
    
    def editing_one_pass(self, S, I, M, l, level, VGGfeatures, netF=None):
        label = (self.max_dilate-1.0) * l + 1.0
        l = torch.tensor(l).float().expand(S.size(0), 1)
        l = l.cuda() if self.gpu else l
        S_in, I_in, S_gt, I_gt, M, feature = self.prepare_training_images(S, I, l, label, level, M)
        L_D = self.update_editing_discriminator(S_in, I_in, S_gt, I_gt, M, l, level, feature)
        L_Gadv, L_perc, L_rec = self.update_editing_generator(S_in, I_in, S_gt, I_gt, M, l, level, VGGfeatures, feature, netF)
        return [L_D, L_Gadv, L_perc, L_rec]
