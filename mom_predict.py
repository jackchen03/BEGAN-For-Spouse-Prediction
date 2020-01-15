import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from collections import deque
import perceptual_models
from perceptual_models import dist_model as dm

import torchvision.utils as vutils

import os
import os.path

from dataloader import *
from models import *
import argparse

import random

parser = argparse.ArgumentParser()
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--ngpu', default=1, type=int)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--b_size', default=16, type=int)
parser.add_argument('--h', default=64, type=int)
parser.add_argument('--nc', default=64, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--lr_update_step', default=3000, type=int)
parser.add_argument('--lr_update_type', default=1, type=int)
parser.add_argument('--lr_lower_boundary', default=2e-6, type=float)
parser.add_argument('--gamma', default=0.5, type=float)
parser.add_argument('--lambda_k', default=0.001, type=float)
parser.add_argument('--k', default=0, type=float)
parser.add_argument('--scale_size', default=64, type=int)
parser.add_argument('--model_name', default='test2')
parser.add_argument('--base_path', default='./')
parser.add_argument('--data_path', default='../dataset/img_align_celeba_128_crop/')
parser.add_argument('--dec_load_path', default='')
parser.add_argument('--load_step', default=0, type=int)
parser.add_argument('--print_step', default=1000, type=int)
parser.add_argument('--num_workers', default=12, type=int)
parser.add_argument('--l_type', default=1, type=int)
parser.add_argument('--tanh', default=1, type=int)
parser.add_argument('--manualSeed', default=5451, type=int)
parser.add_argument('--train', default=1, type=int)
opt = parser.parse_args()


print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.cuda:
    opt.cuda = True
    torch.cuda.set_device(opt.gpuid)
    torch.cuda.manual_seed_all(opt.manualSeed)


class Mom_predict():
    def __init__(self):
        self.global_step = opt.load_step
        self.prepare_paths()
        self.data_loader = get_loader(self.data_path, opt.b_size, opt.scale_size, opt.num_workers)

        self.build_model() 

        # self.z = Variable(torch.FloatTensor(opt.b_size, opt.h))
        # self.fixed_z = Variable(torch.FloatTensor(opt.b_size, opt.h))
        # self.fixed_z.data.uniform_(-1, 1)    
        # self.fixed_x = None
        self.criterion_perceptual = perceptual_models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True, spatial=True)
        self.criterion_l2 = nn.MSELoss()

        if opt.cuda:
            self.set_cuda()

    def set_cuda(self):
        self.De.cuda()
        self.En.cuda()
        # self.z = self.z.cuda()
        # self.fixed_z = self.fixed_z.cuda()
        self.criterion_l2.cuda()
        self.criterion_perceptual.cuda()
 
    def prepare_paths(self):
        self.data_path = opt.data_path
        self.Dec_load_path = opt.dec_load_path
        self.En_save_path = os.path.join(opt.base_path, '%s/models'%opt.model_name)
        self.sample_dir = os.path.join(opt.base_path,  '%s/samples'%opt.model_name)

        for path in [self.En_save_path, self.sample_dir]:
            if not os.path.exists(path):
                os.makedirs(path)
        print("Generated samples saved in %s"%self.sample_dir)
    
    def build_model(self):
        self.En = Encoder(opt)
        self.De = Decoder(opt)
        print (self.En)
        print ("====================")
        print (self.De)
        #disc.apply(weights_init)
        #gen.apply(weights_init)
        step = opt.load_step
        self.De.load_state_dict(torch.load(self.Dec_load_path)) 
        self.De.eval()

        if opt.load_step > 0:
            self.load_models(opt.load_step)

    def generate(self, input, target, sample, step, nrow=8):
        #sample = self.gen(fake)
        #print sample.size()
        #return
        #sample = sample.data.cpu().mul(0.5).add(0.5).mul(255).byte().transpose(0,2).transpose(0,1).numpy()
        #from PIL import Image
        #print type(sample)
        #im = Image.fromarray(sample.astype('uint8'))
        #im.save('128.png')
        vutils.save_image(input.data, '%s/%s_%s_son.png'%(self.sample_dir, opt.model_name, str(step)), nrow=nrow, normalize=True)
        vutils.save_image(target.data, '%s/%s_%s_mom_target.png'%(self.sample_dir, opt.model_name, str(step)), nrow=nrow, normalize=True)
        vutils.save_image(sample.data, '%s/%s_%s_mom_pred.png'%(self.sample_dir, opt.model_name, str(step)), nrow=nrow, normalize=True)
        #f = open('%s/%s_gen.mat'%(self.sample_dir, opt.model_name), 'w')
        #np.save(f, sample.data.cpu().numpy())
        #recon = self.disc(self.fixed_x)
        # if recon is not None:
        #     vutils.save_image(recon.data, '%s/%s_%s_disc.png'%(self.sample_dir, opt.model_name, str(step)), nrow=nrow, normalize=True)

    def save_models(self, step):
        torch.save(self.En.state_dict(), os.path.join(self.En_save_path, 'En_%d.pth'%step)) 
        # torch.save(self.De.state_dict(), os.path.join(self.De_save_path, 'De_%d.pth'%step)) 

    def load_models(self, step):
        self.En.load_state_dict(torch.load(os.path.join(self.En_save_path, 'En_%d.pth'%step)))     

    # def compute_disc_loss(self, outputs_d_x, data, outputs_d_z, gen_z):
    #     if opt.l_type == 1:
    #         real_loss_d = torch.mean(torch.abs(outputs_d_x - data))
    #         fake_loss_d = torch.mean(torch.abs(outputs_d_z - gen_z))
    #     else:
    #         real_loss_d = self.criterion(outputs_d_x, data)
    #         fake_loss_d = self.criterion(outputs_d_z , gen_z.detach())
    #     return (real_loss_d, fake_loss_d)
            
    # def compute_gen_loss(self, outputs_g_z, gen_z):
    #     if opt.l_type == 1:
    #         return torch.mean(torch.abs(outputs_g_z - gen_z))
    #     else:
    #         return self.criterion(outputs_g_z, gen_z)

    def train(self):
        opti = torch.optim.Adam(self.En.parameters(), betas=(0.5, 0.999), lr=opt.lr)
        # measure_history = deque([0]*opt.lr_update_step, opt.lr_update_step)

        # convergence_history = []
        # prev_measure = 1

        lr = opt.lr

        for i in range(opt.epochs):
            print("epoch " + str(i+1) + " starts.")
            for _, (target_mom, input_son) in enumerate(self.data_loader):
                # data = Variable(data)

                if opt.cuda:
                    input_son = input_son.cuda()
                    target_mom = target_mom.cuda()

                m, sigma = self.En(input_son)
                e = torch.randn(input_son.shape[0], 64).cuda()
                encoded_feat = torch.exp(sigma)*e + m
                pred_mom = self.De(encoded_feat)

                opti.zero_grad()
                kl_loss =  ( torch.exp(sigma) - (torch.ones(input_son.shape[0], 64).cuda() + sigma) + m**2 ).mean()
                loss = self.criterion_perceptual(pred_mom, target_mom).mean() + self.criterion_l2(pred_mom, target_mom) + self.criterion_perceptual(pred_mom, input_son).mean() + kl_loss * 5000
                # loss = self.criterion_l1(pred_mom, target_mom)
                # loss = self.criterion_perceptual(pred_mom, target_mom).mean()  + self.criterion_perceptual(pred_mom, input_son).mean()
                loss.backward()
                opti.step()

                if self.global_step%opt.print_step == 0:
                    print ( "Step: %d, Epochs: %d, Loss E: %.9f, KL Loss: %.9f, lr:%.9f"% (self.global_step, i, loss.item(), 5000*kl_loss.item(), lr) )
                    self.generate(input_son, target_mom, pred_mom, self.global_step)
               
                if opt.lr_update_type == 1:
                    lr = opt.lr* 0.95 ** (self.global_step//opt.lr_update_step)
                elif opt.lr_update_type == 2:
                    if self.global_step % opt.lr_update_step == opt.lr_update_step -1 :
                        lr *= 0.5
                elif opt.lr_update_type == 3:
                    if self.global_step % opt.lr_update_step == opt.lr_update_step -1 :
                        lr = min(lr*0.5, opt.lr_lower_boundary)
                else:
                    if self.global_step % opt.lr_update_step == opt.lr_update_step - 1:
                        cur_measure = np.mean(measure_history)
                        if cur_measure > prev_measure * 0.9999:
                            lr = min(lr*0.5, opt.lr_lower_boundary)
                        prev_measure = cur_measure

                if self.global_step%1000 == 0:
                    self.save_models(self.global_step)
            
                #convergence_history.append(convg_measure)
                self.global_step += 1

def generative_experiments(obj):
    z = []
    for inter in range(10):
        z0 = np.random.uniform(-1,1,opt.h)
        z10 = np.random.uniform(-1,1,opt.h)
        def slerp(val, low, high):
            omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
            so = np.sin(omega)
            if so == 0:
                return (1.0-val) * low + val * high # L'Hopital's rule/LERP
            return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high 

        z.append(z0)
        for i in range(1, 9):
            z.append(slerp(i*0.1, z0, z10))
        z.append(z10.reshape(1, opt.h)) 
    z = [_.reshape(1, opt.h) for _ in z]
    z_var = Variable(torch.from_numpy(np.concatenate(z, 0)).float())
    print( z_var.size() )
    if opt.cuda:
        z_var = z_var.cuda()
    gen_z = obj.gen(z_var)
    obj.generate(gen_z, None, 'gen_1014_slerp_%d'%opt.load_step, 10)

    '''
    # Noise arithmetic 
    for i in range(5):
        sum_z = z[i] + z
        z_var = Variable(torch.from_numpy(np.concatenate(z, 0)).float())
        print z_var.size()
        if opt.cuda:
            z_var = z_var.cuda()
        gen_z = obj.gen(z_var)
        obj.generate(gen_z, None, 'gen_1014_slerp_%d'%i)
    '''
           
            
        

if __name__ == "__main__":
    obj = Mom_predict()
    if opt.train:
        obj.train()
    else:
        generative_experiments(obj)
