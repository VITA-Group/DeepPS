from __future__ import print_function

import torch
from options import TrainOptions
from models import PSGAN
from utils import to_var, to_data, visualize, load_image, save_image, get_mask, weights_init
from pix2pix import Pix2pix256, Pix2pix128, Pix2pix64
from vgg import VGGFeature
import random

import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as dset

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    # parse options
    parser = TrainOptions()
    opts = parser.parse()
    
    if opts.model_task != 'SYN' and opts.model_task != 'EDT':
        print('%s:unsupported task!'%opts.model_task)
        return
    SYN = opts.model_task == 'SYN'
    
    # create model
    print('--- create model ---')
    G_channels = 3 if SYN else 7 # SYN: (S), EDT:(S,I,M)
    D_channels = 6 if SYN else 7 # SYN: (S,I), EDT:(S,I,M)
    # (img_size, max_level) should be (256, 3), (128, 2) or (64, 1)
    netG = PSGAN(G_channels, opts.G_nlayers, opts.G_nf, D_channels, opts.D_nf, 
                 opts.D_nlayers, opts.max_dilate, opts.max_level, opts.img_size, opts.gpu!=0)
    if opts.gpu:
        netG.cuda()
    netG.init_networks(weights_init)
    netG.train()
    device = None if opts.gpu else torch.device('cpu')
    
    # to adapt G to the pretrained F
    if opts.use_F_level in [1,2,3]:
        # you could change the setting of netF based on your own pretraining setting 
        netF_Norm = 'None' if SYN else 'BN' 
        if opts.use_F_level == 1:
            netF = Pix2pix64(in_channels = G_channels, nef=64, useNorm=netF_Norm)
        elif opts.use_F_level == 2:
            netF = Pix2pix128(in_channels = G_channels, nef=64, useNorm=netF_Norm)
        else:
            netF = Pix2pix256(in_channels = G_channels, nef=64, useNorm=netF_Norm)
        netF.load_state_dict(torch.load(opts.load_F_name, map_location=device))
        for param in netF.parameters():
            param.requires_grad = False
        if opts.gpu:
            netF = netF.cuda()
    
    # for perceptual loss
    VGGNet = models.vgg19(pretrained=True).features
    VGGfeatures = VGGFeature(VGGNet, opts.gpu)
    for param in VGGfeatures.parameters():
        param.requires_grad = False
    if opts.gpu:
        VGGfeatures.cuda()

    print('--- training ---')
    dataset = dset.ImageFolder(root=opts.train_path,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    
    # sampling refinement level l from l_sample_range
    # we found that r=0.5 should be trained to discriminate the results from those under r=0.0 
    l_sample_range = [1.*r/(opts.max_dilate-1) for r in list(range(opts.max_dilate))+[0.5]]
    
    # We train 64*64 image (max_level = 1) by directly training netG.G64 on 64*64. 
    # We train 256*256 images (max_level = 3) by first training netG.G64 on 64*64,
    # then fixing netG.G64 and training netG.G128 on 128*128,
    # finally fixing netG.G128 and training netG.G256 on 256*256, like pix2pixHD
    
    # NOTE: using large batch size at level1,2 (small image resolution) could improve results
    # However, this could make CUDA out of memory when training goes to level3.
    # Because the memory used in level1,2 is not released.
    # Solution is to call train.py three times, 
    # each time only training one level and saving the model parameters to load for the next level
    # for example, 
    # 'for level in range(1,1+opts.max_level)' should be changed to 'for level in [cur_level]'
    # and when cur_level=2, adding 'netG.G64.load_state_dict(torch.load(saved model at level 1))'
    # when cur_level=3, adding 'netG.G128.load_state_dict(torch.load(saved model at level 2))'
    batchsize_level = [opts.batchsize_level1, opts.batchsize_level2, opts.batchsize_level3]
    # progressively training from level1 to max_level
    for level in range(1,1+opts.max_level):
        print('--- Training at level %d. Image resolution: %d ---' % (level, 2**(5+level)))
        # fix the model parameters
        if level in [2,3]:
            for param in netG.G64.parameters():
                param.requires_grad = False
        if level in [3]:
            for param in netG.G128.parameters():
                param.requires_grad = False
        dataloader = DataLoader(dataset, batch_size=batchsize_level[level-1], shuffle=True, num_workers=4, drop_last=True)
        print_step = int(len(dataloader) / 10.0)
        if not SYN:
            # generate random 2**(13-level) masks for sampling
            masks_num = 2**(13-level)
            all_masks = get_mask(masks_num, opts.img_size)
            all_masks = to_var(all_masks) if opts.gpu else all_masks
        # main iteration
        for epoch in range(opts.epoch_pre+opts.epoch):
            for i, data in enumerate(dataloader):
                # during pretraining epoches, we only train on the max refinement level l = 1.0
                # then we will train on random l in [0,1]
                l = 1.0 if epoch < opts.epoch_pre else random.choice(l_sample_range) 
                data = to_var(data[0]) if opts.gpu else data[0]
                # if input image is arranged as (S,I) use AtoB = True
                # if input image is arranged as (I,S) use AtoB = False
                if opts.AtoB:
                    S = data[:,:,:,0:opts.img_size]
                    I = data[:,:,:,opts.img_size:opts.img_size*2]
                else:
                    S = data[:,:,:,opts.img_size:opts.img_size*2]  
                    I = data[:,:,:,0:opts.img_size]
                # apply netF loss, this will drastically increase the CUDA memory usuage
                netF_level = netF if level == opts.use_F_level else None
                if SYN:
                    losses = netG.synthesis_one_pass(S, I, l, level, VGGfeatures, netF=netF_level)
                else:
                    M = all_masks[torch.randint(masks_num, (batchsize_level[level-1],))]
                    losses = netG.editing_one_pass(S, I, M, l, level, VGGfeatures, netF=netF_level)
                
                if i % print_step == 0:
                    print('Epoch [%03d/%03d][%04d/%04d]' %(epoch+1, opts.epoch_pre+opts.epoch, i+1,
                                                                       len(dataloader)), end=': ')
                    print('l: %+.3f, LD: %+.3f, LGadv: %+.3f, Lperc: %+.3f, Lrec: %+.3f'%
                          (l, losses[0], losses[1], losses[2], losses[3]))
        print('--- Saving model at level %d ---' % level)
        netG.save_model(opts.save_model_path, opts.save_model_name, level=[level])
    
if __name__ == '__main__':
    main()
