from __future__ import print_function

import torch
from options import TestOptions
from models import PSGAN
from utils import to_var, to_data, visualize, load_image, save_image
from pix2pix import Pix2pix256

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    # parse options
    parser = TestOptions()
    opts = parser.parse()

    if opts.model_task != 'SYN' and opts.model_task != 'EDT':
        print('%s:unsupported task!'%opts.model_task)
        return

    SYN = opts.model_task == 'SYN'
    
    # Note: this code is used for 256*256 images
    # To test on 64*64 images, change 'img_size = 256' to 'img_size = 64'
    # change 'netG.load_generator(filepath=opts.model_path, filename=opts.model_name)'
    # to 'netG.G64.load_state_dict(torch.load(your saved 64*64 model, map_location=device))'
    # In models.py, function forward_editing() and forward_synthesis()
    # change 'feature64 = self.G64.forward_feature(input64, l).detach()' 
    # to 'outputs = self.G64(input256, l).detach()' and use outputs for final output
    img_size = 256

    # data loader
    print('----load data----')
    img = to_var(load_image(opts.input_name)) if opts.gpu else load_image(opts.input_name)
    level = opts.l
    step = max(opts.l_step, 0.1) if level == -1 else 100.0
    l = 0.0 if level == -1 else min(max(level, 0.0), 1.0)
    
    if SYN:
        S = img
    else:
        S = img[:,:,:,img_size:img_size*2]
        M = img[:,0:1,:,img_size*2:img_size*3]
        I = img[:,:,:,0:img_size]

    print('----load model----')
    G_channels = 3 if SYN else 7
    netF_Norm = 'None' if SYN else 'BN'
    device = None if opts.gpu else torch.device('cpu')
    netG = PSGAN(G_channels = G_channels, max_dilate = opts.max_dilate, img_size = img_size, gpu = opts.gpu!=0)
    netG.load_generator(filepath=opts.model_path, filename=opts.model_name)
    netF = Pix2pix256(in_channels = G_channels, nef=64, useNorm=netF_Norm)
    netF.load_state_dict(torch.load(opts.load_F_name, map_location=device))
    if opts.gpu:
        netG.cuda()
        netF.cuda()
    netG.eval()
    netF.eval()

    print('----testing----')
    S_gens = []
    I_gens = []
    I_outs = []
    while l <= 1.0:
        if SYN:
            S_gen, I_gen = netG.forward_synthesis(S, l)
            I_out = netF(S_gen).detach()
        else:
            S_gen, I_gen = netG.forward_editing(S, I, M, l)
            I_out = netF(torch.cat((S_gen, I*(1-M)/2, M),dim=1)).detach()
        S_gens += [to_data(S_gen) if opts.gpu else S_gen] 
        I_gens += [to_data(I_gen) if opts.gpu else I_gen]
        I_outs += [to_data(I_out) if opts.gpu else I_out]            
        l += step

    print('----save----')
    if not os.path.exists(opts.result_dir):
        os.mkdir(opts.result_dir)         
    for i in range(len(S_gens)):     
        result_filename = os.path.join(opts.result_dir, opts.name)
        save_image(S_gens[i][0], result_filename+'-SGEN_'+str(i)+'.png')
        save_image(I_gens[i][0], result_filename+'-IGEN_'+str(i)+'.png')
        save_image(I_outs[i][0], result_filename+'-IOUT_'+str(i)+'.png')

if __name__ == '__main__':
    main()