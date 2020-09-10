from __future__ import print_function

import torch
import torch.nn.functional as F
import torch.nn as nn
from options import TrainPix2PixOptions
from models import PSGAN
from utils import to_var, to_data, visualize, load_image, save_image, get_mask, weights_init
from pix2pix import Pix2pix256, DiscriminatorSN
from vgg import VGGFeature
import random

import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as dset
from roughSketchSyn import MyDilateBlur

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# this code is for 256*256 images
def main():
    # parse options
    parser = TrainPix2PixOptions()
    opts = parser.parse()
    
    if opts.model_task != 'SYN' and opts.model_task != 'EDT':
        print('%s:unsupported task!'%opts.model_task)
        return
    SYN = opts.model_task == 'SYN'
    
    # create model
    print('--- create model ---')
    G_channels = 3 if SYN else 7 # SYN: (S), EDT:(S,I,M)
    D_channels = 6 if SYN else 7 # SYN: (S,I), EDT:(S,I,M)
    netF = Pix2pix256(nef=opts.G_nf, out_channels=3, in_channels=G_channels)
    netD = DiscriminatorSN(in_channels=D_channels, out_channels=opts.D_nf, ndf=opts.D_nf, n_layers=opts.D_nlayers, input_size=opts.img_size)
    edgeSmooth = MyDilateBlur()
    
    if opts.gpu:
        netF.cuda()
        netD.cuda()
        edgeSmooth.cuda()
    netF.apply(weights_init)
    netD.apply(weights_init)
    netF.train()
    netD.train()
    
    trainerF = torch.optim.Adam(netF.parameters(), lr=0.0002, betas=(0.5, 0.999))
    trainerD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    device = None if opts.gpu else torch.device('cpu')
    
    # for perceptual loss
    VGGNet = models.vgg19(pretrained=True).features
    VGGfeatures = VGGFeature(VGGNet, opts.gpu)
    for param in VGGfeatures.parameters():
        param.requires_grad = False
    if opts.gpu:
        VGGfeatures.cuda()
        
    L1loss = nn.L1Loss()
    L2loss = nn.MSELoss()

    print('--- training ---')
    dataset = dset.ImageFolder(root=opts.train_path,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    
    dataloader = DataLoader(dataset, batch_size=opts.batchsize, shuffle=True, num_workers=4, drop_last=True)
    
    
    print_step = int(len(dataloader) / 10.0)
    if not SYN:
        # generate random 1024 masks for sampling
        masks_num = 1024
        all_masks = get_mask(masks_num, opts.img_size)
        all_masks = to_var(all_masks) if opts.gpu else all_masks
        
    # main iteration
    for epoch in range(opts.epoch):
        for i, data in enumerate(dataloader):
            data = to_var(data[0]) if opts.gpu else data[0]
            # if input image is arranged as (S,I) use AtoB = True
            # if input image is arranged as (I,S) use AtoB = False
            if opts.AtoB:
                S = data[:,:,:,0:opts.img_size]
                I = data[:,:,:,opts.img_size:opts.img_size*2]
            else:
                S = data[:,:,:,opts.img_size:opts.img_size*2]  
                I = data[:,:,:,0:opts.img_size]
            
            # train netD
            S = edgeSmooth(S)
            if not SYN:
                M = to_var(all_masks[torch.randint(1024, (opts.batchsize,))])
                S = S * (1+M)/2
                I_in = I * (1-M)/2
            real_input = S if SYN else torch.cat((S, I_in, M), dim=1)
            
            real_concat = torch.cat((S, I), dim=1)
            with torch.no_grad():
                fake_concat = torch.cat((S,netF(real_input)), dim=1)
            if not SYN:
                real_concat = torch.cat((real_concat,M), dim=1)
                fake_concat = torch.cat((fake_concat,M), dim=1)
            real_output = netD(real_concat)
            fake_output = netD(fake_concat)    
            L_D = opts.weight_adv*((F.relu(opts.hinge-real_output)).mean() + 
                               (F.relu(opts.hinge+fake_output)).mean())
            trainerD.zero_grad()
            L_D.backward()
            trainerD.step()
            
            
            # train netF
            with torch.no_grad():
                real_Phi = VGGfeatures(I)
            fake_concat = torch.cat((S,netF(real_input)), dim=1)
            if not SYN:
                fake_concat = torch.cat((fake_concat,M), dim=1)
            fake_output = netD(fake_concat)
            L_Gadv = -opts.weight_adv*fake_output.mean()        
            fake_Phi = VGGfeatures(fake_concat[:,3:6])
            L_perc = sum([opts.weight_perc[ii] * L2loss(A, real_Phi[ii]) for ii,A in enumerate(fake_Phi)])
            L_rec = opts.weight_rec * L1loss(fake_concat, real_concat)

            L_F = L_Gadv + L_perc + L_rec

            trainerF.zero_grad()
            L_F.backward()
            trainerF.step()


            if i % print_step == 0:
                print('Epoch [%03d/%03d][%04d/%04d]' %(epoch+1, opts.epoch, i+1,
                                                                   len(dataloader)), end=': ')
                print('LD: %+.3f, LGadv: %+.3f, Lperc: %+.3f, Lrec: %+.3f'%
                      (L_D.data.mean(), L_Gadv.data.mean(), L_perc.data.mean(), L_rec.data.mean()))
    print('--- Saving model ---')
    torch.save(netF.state_dict(), os.path.join(opts.save_model_path, opts.save_model_name+'-F256.ckpt'))
    torch.save(netD.state_dict(), os.path.join(opts.save_model_path, opts.save_model_name+'-FD256.ckpt'))
    
if __name__ == '__main__':
    main()