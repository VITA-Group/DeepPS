import torch

from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import cv2
import scipy.ndimage as pyimg
import os
import random

from torch.utils import data
from torchvision import transforms as T
from torchsample.transforms import Rotate

def to_var(x):
    """Convert tensor to variable."""
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def to_data(x):
    """Convert variable to tensor."""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data

# custom weights initialization called on networks
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if not(m.bias is None):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# view images
def visualize(img_arr):
    plt.imshow(((img_arr.numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
    plt.axis('off')

# load one image in tensor format
def load_image(filename, load_type=0, wd=256, ht=256):
    #centerCrop = transforms.CenterCrop((wd, ht))
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5]),
    ])
    
    if load_type == 0:
        img = transform(Image.open(filename))
    else:
        img = transform(text_image_preprocessing(filename))
        
    return img.unsqueeze(dim=0)

def save_image(img, filename):
    tmp = ((img.numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
    cv2.imwrite(filename, cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR))
    
# generate random masks for image editing training  
def get_mask(batch_size, image_size):
    min_ht = image_size // 4
    max_ht = min_ht * 3
    masks = torch.zeros(batch_size,1,image_size,image_size)
    for i in range(batch_size):
        mask_w = np.random.randint(min_ht, max_ht)
        mask_h = np.random.randint(min_ht, max_ht)
        px = np.random.randint(0, image_size-mask_w)
        py = np.random.randint(0, image_size-mask_h)
        pr = np.random.randint(0, 45)
        mask = torch.zeros(1, image_size, image_size)
        mask[:,py:py+mask_h, px:px+mask_w] = 1
        mask = Rotate(pr)(mask)
        masks[i] = mask
    unknown = masks > 0.5
    masks[unknown] = 1
    masks[~unknown] = -1
    return masks

# binarize an image
def binarize(input, threshold=0):
    mask = input > threshold
    input[mask] = 1
    input[~mask] = -1
    return input
