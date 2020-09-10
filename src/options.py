import argparse

class TestOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # data loader related
        self.parser.add_argument('--input_name', type=str, default='../data/SYN/5.png', help='path of the style image')
        self.parser.add_argument('--model_task', type=str, default='SYN', help='SYN for image synthesis, EDT for image editing')

        # ouptput related
        self.parser.add_argument('--name', type=str, default='output', help='file name of the outputs')
        self.parser.add_argument('--result_dir', type=str, default='../output/', help='path for saving result images')
        self.parser.add_argument('--l', type=float, default=-1, help='refinement level,  0~1 for single level, -1 for multiple level in 0~1 with step of scale_step')
        self.parser.add_argument('--l_step', type=float, default=0.25, help='level step')

        # model related
        self.parser.add_argument('--model_path', type=str, default='../save/', help='specify the model path to load')
        self.parser.add_argument('--model_name', type=str, default='ECCV-SYN-celebaHQ', help='specify the model name to load')
        self.parser.add_argument('--load_F_name', type=str, default='../save/ECCV-SYN-celebaHQ-F256.ckpt', help='specified the dir of saved network F')
        self.parser.add_argument('--gpu', type=int, default=1, help='gpu, 0 for cpu, 1 for gpu')
        self.parser.add_argument('--max_dilate', type=int, default=21, help='max dilation diameter')

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('--- load options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt
    
class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # data loader related
        self.parser.add_argument('--train_path', type=str, default='../data/dataset/', help='path of the training images')
        self.parser.add_argument('--AtoB', action='store_true', default=False, help='Whether the input images are arranged in form of [S,I]. Otherwise, [I,S]')
        self.parser.add_argument('--img_size', type=int, default=256, help='training image size')     

        # train related
        self.parser.add_argument('--model_task', type=str, default='SYN', help='SYN for image synthesis, EDT for image editing')
        self.parser.add_argument('--epoch_pre', type=int, default=30, help='number of epoch for pretraining on the max refinement level')
        self.parser.add_argument('--epoch', type=int, default=200, help='number of epoch')       
        self.parser.add_argument('--batchsize_level1', type=int, default=64, help='batchsize for level1.')
        self.parser.add_argument('--batchsize_level2', type=int, default=32, help='batchsize for level2.')
        self.parser.add_argument('--batchsize_level3', type=int, default=4, help='batchsize for level3.')        
        
        self.parser.add_argument('--weight_rec', type=float, default=100.0, help='weight for reconstruction losses')
        self.parser.add_argument('--weight_adv', type=float, default=1.0, help='weight for adversarial losses')
        self.parser.add_argument('--weight_perc', nargs='+', type=float, default=[1.0,0.5], help='weights for perceptual losses on conv2_1 and conv3_1')        
        self.parser.add_argument('--hinge', type=float, default=10.0, help='margin parameter for hinge loss')
        self.parser.add_argument('--max_level', type=int, default=3, help='specify the highest level for training. Level 1, 2, 3 for images at resoluiton 64, 128, 256.') 

        # model related
        self.parser.add_argument('--G_nlayers', type=int, default=3, help='number of ResnetBlock layers in generator G')
        self.parser.add_argument('--D_nlayers', type=int, default=3, help='number of layers in discriminator D')  
        self.parser.add_argument('--G_nf', type=int, default=64, help='number of features in the first layer of G')
        self.parser.add_argument('--D_nf', type=int, default=64, help='number of features in the first layer of D') 
        self.parser.add_argument('--max_dilate', type=int, default=21, help='max dilation diameter')
        self.parser.add_argument('--save_model_path', type=str, default='../save/', help='specify the model path to save')        
        self.parser.add_argument('--save_model_name', type=str, default='PSGAN', help='specify the model name to save')
        self.parser.add_argument('--gpu', type=int, default=1, help='gpu, 0 for cpu, 1 for gpu')
        self.parser.add_argument('--load_F_name', type=str, default='../save/ECCV-SYN-celebaHQ-F256.ckpt', help='specified the dir of saved network F')
        self.parser.add_argument('--use_F_level', type=int, default=3, help='specify the level to use network F for loss calculation')
        
    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('--- load options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt
    
class TrainPix2PixOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # data loader related
        self.parser.add_argument('--train_path', type=str, default='../data/dataset/', help='path of the training images')
        self.parser.add_argument('--AtoB', action='store_true', default=False, help='Whether the input images are arranged in form of [S,I]. Otherwise, [I,S]')
        self.parser.add_argument('--img_size', type=int, default=256, help='training image size')     

        # train related
        self.parser.add_argument('--model_task', type=str, default='SYN', help='SYN for image synthesis, EDT for image editing')
        self.parser.add_argument('--epoch', type=int, default=50, help='number of epoch')       
        self.parser.add_argument('--batchsize', type=int, default=4, help='batchsize')
        
        self.parser.add_argument('--weight_rec', type=float, default=100.0, help='weight for reconstruction losses')
        self.parser.add_argument('--weight_adv', type=float, default=1.0, help='weight for adversarial losses')
        self.parser.add_argument('--weight_perc', nargs='+', type=float, default=[1.0,0.5], help='weights for perceptual losses on conv2_1 and conv3_1')        
        self.parser.add_argument('--hinge', type=float, default=10.0, help='margin parameter for hinge loss')

        # model related
        self.parser.add_argument('--D_nlayers', type=int, default=5, help='number of layers in discriminator D')  
        self.parser.add_argument('--G_nf', type=int, default=64, help='number of features in the first layer of G')
        self.parser.add_argument('--D_nf', type=int, default=64, help='number of features in the first layer of D') 
        self.parser.add_argument('--save_model_path', type=str, default='../save/', help='specify the model path to save')        
        self.parser.add_argument('--save_model_name', type=str, default='PSGAN', help='specify the model name to save')
        self.parser.add_argument('--gpu', type=int, default=1, help='gpu, 0 for cpu, 1 for gpu')
        
    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('--- load options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt
