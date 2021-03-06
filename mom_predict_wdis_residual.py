import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from collections import deque
import perceptual_models
from perceptual_models import dist_model as dm

import torchvision.utils as vutils
import torchvision.models as models

import os
import os.path

from dataloader import *
from models import *
import argparse

import random

from vgg import vgg16
from torch.utils.model_zoo import load_url as load_state_dict_from_url

parser = argparse.ArgumentParser()
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--ngpu', default=1, type=int)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--b_size', default=16, type=int)
parser.add_argument('--h', default=64, type=int)
parser.add_argument('--nc', default=64, type=int)
parser.add_argument('--epochs', default=2000, type=int)
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
parser.add_argument('--data_path', default='../dataset/FID_64_repli_align/')
parser.add_argument('--dec_load_path', default='celebA_64multi_pixelshuffle/experiments/test2/models/gen_328000.pth') 
parser.add_argument('--dis_load_path', default='celebA_64multi_pixelshuffle/experiments/test2/models/disc_328000.pth')
parser.add_argument('--load_step', default=0, type=int)
parser.add_argument('--print_step', default=1000, type=int)
parser.add_argument('--num_workers', default=12, type=int)
parser.add_argument('--l_type', default=1, type=int)
parser.add_argument('--tanh', default=1, type=int)
parser.add_argument('--manualSeed', default=5451, type=int)
parser.add_argument('--train', default=1, type=int)
parser.add_argument('--upsample_type', default='pixelcnn', type=str, choices=['pixelcnn', 'nearest'])
parser.add_argument('--g_pretrain', default=False, type=bool)
parser.add_argument('--lambda_g', default=1.0, type=float)
parser.add_argument('--lambda_d', default=1.0, type=float)
parser.add_argument('--lambda_kl', default=100, type=float)
parser.add_argument('--lambda_per', default=0.05, type=float)
parser.add_argument('--lambda_l1', default=4.0, type=float)


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
		self.data_loader = get_loader(self.data_path, opt.b_size, opt.scale_size, True, opt.num_workers)

		self.build_model() 

		# self.z = Variable(torch.FloatTensor(opt.b_size, opt.h))
		# self.fixed_z = Variable(torch.FloatTensor(opt.b_size, opt.h))
		# self.fixed_z.data.uniform_(-1, 1)	
		# self.fixed_x = None

		# self.criterion_perceptual = perceptual_models.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True, spatial=True)
		self.vgg = vgg16(pretrained=True)
		self.vgg.eval()
		self.criterion_l1 = nn.L1Loss()
		self.criterion_l2 = nn.MSELoss()
		self.adversarial_loss = torch.nn.BCELoss()

		if opt.cuda:
			self.set_cuda()

	def set_cuda(self):
		self.vgg.cuda()
		self.De.cuda()
		self.En.cuda()
		self.Dis.cuda()
		# self.z = self.z.cuda()
		# self.fixed_z = self.fixed_z.cuda()
		self.criterion_l1.cuda()
		self.adversarial_loss.cuda()
 
	def prepare_paths(self):
		self.data_path = opt.data_path
		self.Dec_load_path = opt.dec_load_path
		self.Dis_load_path = opt.dis_load_path
		self.En_save_path = os.path.join(opt.base_path, '%s/models'%opt.model_name)
		self.sample_dir = os.path.join(opt.base_path,  '%s/samples'%opt.model_name)

		for path in [self.En_save_path, self.sample_dir]:
			if not os.path.exists(path):
				os.makedirs(path)
		print("Generated samples saved in %s"%self.sample_dir)

	def build_model(self):
		self.En = Spouse_Encoder(opt)
		self.De = Generator(opt)
		self.Dis = Discriminator(opt)

		# self.En.apply(weights_init)
		# self.Dis.apply(weights_init)
		if opt.g_pretrain == True:
			self.De.load_state_dict(torch.load(self.Dec_load_path)) 
		# else: 
		# 	self.De.apply(weights_init)
		# self.De.eval()
		# self.Dis.load_state_dict(torch.load(self.Dis_load_path))
		# self.Dis.eval()

		step = opt.load_step
		if opt.load_step > 0:
			self.load_models(opt.load_step)

	def generate(self, input, target, sample, residual, warped_input, step, nrow=8):
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
		vutils.save_image(residual.data, '%s/%s_%s)_residual.png'%(self.sample_dir, opt.model_name, str(step)), nrow=nrow, normalize=True)
		vutils.save_image(warped_input.data, '%s/%s_%s)_warped_input.png'%(self.sample_dir, opt.model_name, str(step)), nrow=nrow, normalize=True)
		#f = open('%s/%s_gen.mat'%(self.sample_dir, opt.model_name), 'w')
		#np.save(f, sample.data.cpu().numpy())
		#recon = self.disc(self.fixed_x)
		# if recon is not None:
		#	 vutils.save_image(recon.data, '%s/%s_%s_disc.png'%(self.sample_dir, opt.model_name, str(step)), nrow=nrow, normalize=True)


	def save_models(self, step):
		torch.save(self.En.state_dict(), os.path.join(self.En_save_path, 'En_%d.pth'%step)) 
		torch.save(self.De.state_dict(), os.path.join(self.En_save_path, 'De_%d.pth'%step))		 
		# torch.save(self.Dis.state_dict(), os.path.join(self.En_save_path, 'Dis3_%d.pth'%step)) 

	def load_models(self, step):
		self.En.load_state_dict(torch.load(os.path.join(self.En_save_path, 'En_%d.pth'%step)))	 

	def compute_perceptual_loss(self, pred, gt):
		layers = []
		layers_1 = self.vgg.inter_layer(5)
		layers_2 = self.vgg.inter_layer(10)
		layers_3 = self.vgg.inter_layer(17)
		layers_4 = self.vgg.inter_layer(24)
		layers_5 = self.vgg.inter_layer(31)
		# print(layers_3)
		# layers_4 = torch.nn.Sequential(*list(self.vgg.children())[:3])
		# layers_5 = torch.nn.Sequential(*list(self.vgg.children())[:13])
		# layers_6 = torch.nn.Sequential(*list(self.vgg.children())[:-1])

		dist = 0
		dist += self.criterion_l2(layers_1(pred), layers_1(gt))
		dist += self.criterion_l2(layers_2(pred), layers_2(gt))
		dist += self.criterion_l2(layers_3(pred), layers_3(gt))
		dist += self.criterion_l2(layers_4(pred), layers_4(gt))
		dist += self.criterion_l2(layers_5(pred), layers_5(gt))
		# dist += self.criterion_l2(layers_4(pred), layers_4(gt))
		# dist += self.criterion_l2(layers_5(pred), layers_5(gt))
		# dist += self.criterion_l2(layers_6(pred), layers_6(gt))

		return dist

	def compute_disc_loss(self, real_img, fake_img):
		if opt.cuda:
			valid = torch.ones(real_img.shape[0], 1).cuda()
			fake = torch.zeros(real_img.shape[0], 1).cuda()
		else:
			valid = torch.ones(real_img.shape[0], 1)
			fake = torch.zeros(real_img.shape[0], 1)
		fake_loss = self.adversarial_loss(self.Dis(fake_img), fake)
		real_loss = self.adversarial_loss(self.Dis(real_img), valid)

		return fake_loss + real_loss

	def compute_gen_loss(self, fake_img):
		if opt.cuda:
			valid = torch.ones(fake_img.shape[0], 1).cuda()
		else:
			valid = torch.ones(fake_img.shape[0], 1)

		loss = self.adversarial_loss(self.Dis(fake_img), valid)

		return loss
	# def compute_disc_loss(self, outputs_d_x, data, outputs_d_z, gen_z):
	# 	if opt.l_type == 1:
	# 		real_loss_d = torch.mean(torch.abs(outputs_d_x - data))
	# 		fake_loss_d = torch.mean(torch.abs(outputs_d_z - gen_z))
	# 	else:
	# 		real_loss_d = self.criterion(outputs_d_x, data)
	# 		fake_loss_d = self.criterion(outputs_d_z , gen_z.detach())
	# 	return (real_loss_d, fake_loss_d)
			
	# def compute_gen_loss(self, outputs_g_z, gen_z):
	# 	if opt.l_type == 1:
	# 		return torch.mean(torch.abs(outputs_g_z - gen_z))
	# 	else:
	# 		return self.criterion(outputs_g_z, gen_z)

	def warp(self, x, flo):
		"""
		warp an image/tensor (im2) back to im1, according to the optical flow
		x: [B, C, H, W] (im2)
		flo: [B, 2, H, W] flow
		"""
		B, C, H, W = x.size()
		# mesh grid 
		xx = torch.arange(0, W).view(1,-1).repeat(H,1)
		yy = torch.arange(0, H).view(-1,1).repeat(1,W)
		xx = xx.view(1,1,H,W).repeat(B,1,1,1)
		yy = yy.view(1,1,H,W).repeat(B,1,1,1)
		grid = torch.cat((xx,yy),1).float()

		if x.is_cuda:
			grid = grid.cuda()
		vgrid = Variable(grid) + flo

		# scale grid to [-1,1] 
		vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
		vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

		vgrid = vgrid.permute(0,2,3,1)		
		output = nn.functional.grid_sample(x, vgrid)
		mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
		mask = nn.functional.grid_sample(mask, vgrid)
			
		mask[mask<0.999] = 0
		mask[mask>0] = 1
		
		return output*mask

	def train(self):
		optimizer_En = torch.optim.Adam(self.En.parameters(), betas=(0.5, 0.999), lr=opt.lr)
		optimizer_De = torch.optim.Adam(self.De.parameters(), betas=(0.5, 0.999), lr=opt.lr)
		optimizer_Dis = torch.optim.Adam(self.Dis.parameters(), betas=(0.5, 0.999), lr=opt.lr)
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
				# print(encoded_feat[0])
				# print(encoded_feat[1])
				e = torch.randn(input_son.shape[0], 64).cuda()
				encoded_feat = torch.exp(sigma)*e + m

				img_pred, flow_pred = self.De(encoded_feat)
				warped_son = self.warp(input_son, flow_pred)
				pred_mom = img_pred + warped_son

				# update discriminator
				optimizer_Dis.zero_grad()
				d_loss = opt.lambda_d * self.compute_disc_loss(target_mom, pred_mom.detach())
				d_loss.backward()
				optimizer_Dis.step()

				# update generator
				optimizer_En.zero_grad()
				optimizer_De.zero_grad()

				g_loss = opt.lambda_g * self.compute_gen_loss(pred_mom)				
				kl_loss =  opt.lambda_kl * ( torch.exp(sigma) - (torch.ones(input_son.shape[0], 64).cuda() + sigma) + m**2 ).mean()
				# perceptual_loss = 0.05 * self.compute_perceptual_loss(pred_mom, target_mom)
				perceptual_loss = opt.lambda_per * (self.compute_perceptual_loss(pred_mom, target_mom) + self.compute_perceptual_loss(pred_mom, input_son))
				# perceptual_loss = self.criterion_perceptual(pred_mom, target_mom).mean()  + self.criterion_perceptual(pred_mom, input_son).mean()
				l1_loss = opt.lambda_l1 * self.criterion_l1(pred_mom, target_mom)
				loss = g_loss + perceptual_loss + l1_loss + kl_loss 
				# loss = self.criterion_l1(pred_mom, target_mom)
				loss.backward()

				optimizer_En.step()
				optimizer_De.step()
				

				if self.global_step%opt.print_step == 0:
					print ( 
						"Step: %d, Epochs: %d, Loss : %.9f, KL Loss: %.9f, Per Loss: %.9f, Recon Loss: %.9f, Gen Loss: %.9f, Dis Loss: %.9f, lr:%.9f"% 
						(self.global_step, i, loss.item(), kl_loss.item(), perceptual_loss.item(), l1_loss.item(), g_loss.item(), d_loss.item(), lr) 
					)
					self.generate(input_son, target_mom, pred_mom, img_pred, warped_son, self.global_step)
			   
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