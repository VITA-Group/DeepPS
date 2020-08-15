import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm2d, LeakyReLU, ConvTranspose2d, ReLU, Tanh, InstanceNorm2d, ReplicationPad2d
import torch.nn.functional as F
import random
from utils import load_image, to_data, to_var

# for computing perceptual loss
class VGGFeature(nn.Module):
    def __init__(self, cnn, gpu=True):
        super(VGGFeature, self).__init__()
        
        self.model1 = cnn[:2]
        self.model2 = cnn[2:7]
        self.model3 = cnn[7:12]
        cnn_normalization_mean = (torch.tensor([0.485, 0.456, 0.406]) + 1) * 0.5
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]) * 0.5
        self.cnn_normalization_mean = cnn_normalization_mean.view(-1, 1, 1)
        self.cnn_normalization_std = cnn_normalization_std.view(-1, 1, 1)
        if gpu:
            self.cnn_normalization_mean = self.cnn_normalization_mean.cuda()
            self.cnn_normalization_std = self.cnn_normalization_std.cuda()
        
    def forward(self, x):
        x = (x - self.cnn_normalization_mean) / self.cnn_normalization_std
        conv1_1 = self.model1(x)
        conv2_1 = self.model2(conv1_1)
        conv3_1 = self.model3(conv2_1)
        return [conv2_1, conv3_1]
