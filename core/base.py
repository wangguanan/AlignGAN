import os
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import itertools

from .model import Generator, Discriminator, ConditionalDiscriminator, weights_init_normal, Encoder, Embeder
from tools import *


class Base:

	def __init__(self, config, loaders):

		self.config = config

		# paths
		self.loaders = loaders
		self.save_path = config.save_path
		self.save_model_path = os.path.join(self.save_path, 'model/')
		self.save_features_path = os.path.join(self.save_path, 'features/')
		self.save_logs_path = os.path.join(self.save_path, 'logs/')
		self.save_results_path = os.path.join(self.save_path, 'results/')
		self.save_images_path = os.path.join(self.save_path, 'images/')


		# dataset configuration
		self.class_num = config.class_num
		self.part_num = 1

		# pixel restore configuration
		self.G_rgb2ir_restore_path = config.G_rgb2ir_restore_path
		self.G_ir2rgb_restore_path = config.G_ir2rgb_restore_path
		self.D_ir_restore_path = config.D_ir_restore_path
		self.D_rgb_restore_path = config.D_rgb_restore_path
		# feature restore configuration
		self.encoder_restore_path = config.encoder_restore_path
		self.embeder_restore_path = config.embeder_restore_path

		# [pixel align part] criterion configuration
		self.train_pixel_part = config.train_pixel_part
		self.lambda_pixel_tri = config.lambda_pixel_tri
		self.lambda_pixel_cls = config.lambda_pixel_cls

		# [feature align part] criterion configuration
		self.margin = config.margin
		self.soft_bh = config.soft_bh
		self.lambda_feature_cls = config.lambda_feature_cls
		self.lambda_feature_triplet = config.lambda_feature_triplet
		self.lambda_feature_gan = config.lambda_feature_gan

		# train configuration
		self.base_pixel_learning_rate = config.base_pixel_learning_rate
		self.base_feature_ide_learning_rate = config.base_feature_ide_learning_rate
		self.milestones = config.milestones
		self.device = torch.device('cuda')


		# test configuration
		self.modes = config.modes
		self.number_shots = config.number_shots
		self.matlab = config.matlab

		# inits
		self._init_model()
		self._init_criterions()
		self._init_optimizer()
		self._init_fixed_values()



	def _init_model(self):

		## the pixel alignment module
		self.G_rgb2ir = Generator(64, 4)
		self.D_ir = ConditionalDiscriminator(64)
		self.D_ir_warmup = Discriminator(128, 64, 4)

		self.G_rgb2ir.apply(weights_init_normal)
		self.D_ir.apply(weights_init_normal)
		self.D_ir_warmup.apply(weights_init_normal)

		self.G_rgb2ir = torch.nn.DataParallel(self.G_rgb2ir).to(self.device)
		self.D_ir = torch.nn.DataParallel(self.D_ir).to(self.device)
		self.D_ir_warmup = torch.nn.DataParallel(self.D_ir_warmup).to(self.device)

		self.G_ir2rgb = Generator(64, 4)
		self.D_rgb = Discriminator(128, 64, 4)
		self.D_rgb_warmup = Discriminator(128, 64, 4)

		self.G_ir2rgb.apply(weights_init_normal)
		self.D_rgb.apply(weights_init_normal)
		self.D_rgb_warmup.apply(weights_init_normal)

		self.G_ir2rgb = torch.nn.DataParallel(self.G_ir2rgb).to(self.device)
		self.D_rgb = torch.nn.DataParallel(self.D_rgb).to(self.device)
		self.D_rgb_warmup = torch.nn.DataParallel(self.D_rgb_warmup).to(self.device)


		## the feature alignment module
		encoder = Encoder()
		embeder = Embeder(1, self.class_num)
		self.encoder = torch.nn.DataParallel(encoder).to(self.device)
		self.embeder = torch.nn.DataParallel(embeder).to(self.device)

		## Restore the pixel alignment module, if the pre-trained model is existed
		if self.G_rgb2ir_restore_path is not 'None':
			self.G_rgb2ir.load_state_dict(torch.load(self.G_rgb2ir_restore_path))
			print('Note: Successfully resume G_rgb2ir from {}'.format(self.G_rgb2ir_restore_path))
		if self.G_ir2rgb_restore_path is not 'None':
			self.G_ir2rgb.load_state_dict(torch.load(self.G_ir2rgb_restore_path))
			print('Note: Successfully resume G_ir2rgb from {}'.format(self.G_ir2rgb_restore_path))
		if self.D_ir_restore_path is not 'None':
			self.D_ir.load_state_dict(torch.load(self.D_ir_restore_path))
			print('Note: Successfully resume D_ir from {}'.format(self.D_ir_restore_path))
		if self.D_rgb_restore_path is not 'None':
			self.D_rgb.load_state_dict(torch.load(self.D_rgb_restore_path))
			print('Note: Successfully resume D_rgb from {}'.format(self.D_rgb_restore_path))


		## restore the feature alignment module, if the pre-trained model is existed
		if self.encoder_restore_path is not 'None':
			self.encoder.load_state_dict(torch.load(self.encoder_restore_path))
			print('Note: Successfully resume generator from {}'.format(self.encoder_restore_path))
		if self.embeder_restore_path is not 'None':
			self.embeder.load_state_dict(torch.load(self.embeder_restore_path))
			print('Note: Successfully resume generator from {}'.format(self.embeder_restore_path))

		## we add all models to a list for esay using, such as train, test, save and restore
		self.model_list = []
		self.model_list.append(self.G_rgb2ir)
		self.model_list.append(self.G_ir2rgb)
		self.model_list.append(self.D_rgb)
		self.model_list.append(self.D_ir)
		self.model_list.append(self.D_rgb_warmup)
		self.model_list.append(self.D_ir_warmup)
		self.model_list.append(self.encoder)
		self.model_list.append(self.embeder)



	def _init_criterions(self):
		# of the pixel alignment module
		self.criterion_gan_mse = torch.nn.MSELoss()
		self.criterion_gan_cycle = torch.nn.L1Loss()
		self.criterion_gan_identity = torch.nn.L1Loss()
		self.ones = torch.ones([self.config.p_gan*self.config.k_gan, 1, 4, 4]).float().to(self.device)
		self.zeros = torch.zeros([self.config.p_gan*self.config.k_gan, 1, 4, 4]).float().to(self.device)

		# of the feature alignment module
		self.criterion_ide_cls = nn.CrossEntropyLoss()
		self.criterion_gan_triplet = TripletLoss(self.margin, self.soft_bh, 'cosine')


	def compute_classification_loss(self, logits_list, pids):
		logits_avg = 0
		loss_avg = 0
		for i in range(self.part_num):
			logits_i = logits_list[i]
			logits_avg += 1.0 / float(self.part_num) * logits_i
			loss_i = self.criterion_ide_cls(logits_i, pids)
			loss_avg += 1.0 / float(self.part_num) * loss_i
		acc = accuracy(logits_avg, pids, (1, 5))
		return acc, loss_avg


	def compute_triplet_loss(self, embedding1_list, embedding2_list, embedding3_list, pids1, pids2, pids3):
		loss_avg = 0
		for i in range(self.part_num):
			embbedding1_i = embedding1_list[i]
			embbedding2_i = embedding2_list[i]
			embbedding3_i = embedding3_list[i]
			loss_i = self.criterion_gan_triplet(embbedding1_i, embbedding2_i, embbedding3_i, pids1, pids2, pids3)
			loss_avg += 1.0 / float(self.part_num) * loss_i
		return loss_avg


	def _init_optimizer(self):

		## of the pixel alignment module
		self.G_optimizer = optim.Adam(itertools.chain(self.G_ir2rgb.parameters(), self.G_rgb2ir.parameters()), lr=self.base_pixel_learning_rate, betas=[0.5, 0.999])
		self.D_rgb_optimizer = optim.Adam(itertools.chain(self.D_rgb.parameters(), self.D_rgb_warmup.parameters()), lr=self.base_pixel_learning_rate, betas=[0.5, 0.999])
		self.D_ir_optimizer = optim.Adam(itertools.chain(self.D_ir.parameters(), self.D_ir_warmup.parameters()), lr=self.base_pixel_learning_rate, betas=[0.5, 0.999])

		self.G_lr_decay = optim.lr_scheduler.MultiStepLR(self.G_optimizer, self.milestones, gamma=0.1)
		self.D_rgb_lr_decay = optim.lr_scheduler.MultiStepLR(self.D_rgb_optimizer, self.milestones, gamma=0.1)
		self.D_ir_lr_decay = optim.lr_scheduler.MultiStepLR(self.D_ir_optimizer, self.milestones, gamma=0.1)

		## of the feature alignment module
		params = [{'params': self.encoder.parameters(), 'lr': 0.1 * self.base_feature_ide_learning_rate},
				  {'params': self.embeder.parameters(), 'lr': self.base_feature_ide_learning_rate}]
		self.ide_optimizer = optim.SGD(params=params, weight_decay=5e-4, momentum=0.9, nesterov=True)
		self.ide_lr_scheduler = optim.lr_scheduler.MultiStepLR(self.ide_optimizer, self.milestones, gamma=0.1)


	def _init_fixed_values(self): # for generating fake images

		self.fixed_real_rgb_images, _, _, _ = self.loaders.rgb_train_iter_ide.next_one()
		self.fixed_real_ir_images, _, _, _ = self.loaders.ir_train_iter_ide.next_one()

		self.fixed_real_rgb_images = self.fixed_real_rgb_images.to(self.device)
		self.fixed_real_ir_images = self.fixed_real_ir_images.to(self.device)



	def process_images_4_encoder(self, images, on_gpu, to_gpu):
		'''

		:param images: 128x128, range [-1, 1]
		:param on_gpu:
		:param to_gpu:
		:return:  384x192, normalize with mean and std as below
		'''

		if not on_gpu:
			images = images.to(self.device)

		# resize to [384, 192] for ide input
		images = F.interpolate(images, [384, 192], mode='bilinear')

		# normalize to [0, 1]
		images = (images + 1.0) / 2.0

		# normalize to resnet input
		mean = [0.485, 0.456, 0.406]
		std = [0.229, 0.224, 0.225]
		mean = torch.tensor(mean).view([1, 3, 1, 1]).repeat([images.size(0), 1, images.size(2), images.size(3)]).to(self.device)
		std = torch.tensor(std).view([1, 3, 1, 1]).repeat([images.size(0), 1, images.size(2), images.size(3)]).to(self.device)
		images = (images - mean) / std

		if not to_gpu:
			images = images.cpu()

		return images


	def generate_wrong_images(self, images, pids):

		num = images.size(0)
		wrong_images = torch.cat([images[4:, :, :, :], images[: 4, :, :, :]], dim=0)
		wrong_pids = torch.cat([pids[4:], pids[:4]])

		return wrong_images, wrong_pids


	def lr_decay(self, current_step):
		self.G_lr_decay.step(current_step)
		self.D_rgb_lr_decay.step(current_step)
		self.D_ir_lr_decay.step(current_step)
		self.ide_lr_scheduler.step(current_step)


	def save_model(self, save_epoch):

		# save model
		for ii, _ in enumerate(self.model_list):
			torch.save(self.model_list[ii].state_dict(),
			           os.path.join(self.save_model_path, 'model-{}_{}.pkl'.format(ii, save_epoch)))

		# if saved model is more than max num, delete the model with smallest epoch
		if self.config.max_save_model_num > 0:
			root, _, files = os_walk(self.save_model_path)

			# get indexes of saved models
			indexes = []
			for file in files:
				indexes.append(int(file.replace('.pkl', '').split('_')[-1]))

			# remove the bad-case and get available indexes
			model_num = len(self.model_list)
			available_indexes = copy.deepcopy(indexes)
			for element in indexes:
				if indexes.count(element) < model_num:
					available_indexes.remove(element)

			available_indexes = sorted(list(set(available_indexes)), reverse=True)
			unavailable_indexes = list(set(indexes).difference(set(available_indexes)))

			# delete all unavailable models
			for unavailable_index in unavailable_indexes:
				try:
					# os.system('find . -name "{}*_{}.pkl" | xargs rm  -rf'.format(self.config.save_models_path, unavailable_index))
					for ii in range(len(self.model_list)):
						os.remove(os.path.join(root, 'model-{}_{}.pkl'.format(ii, unavailable_index)))
				except:
					pass

			# delete extra models
			if len(available_indexes) >= self.config.max_save_model_num:
				for extra_available_index in available_indexes[self.config.max_save_model_num:]:
					# os.system('find . -name "{}*_{}.pkl" | xargs rm  -rf'.format(self.config.save_models_path, extra_available_index))
					for ii in range(len(self.model_list)):
						os.remove(os.path.join(root, 'model-{}_{}.pkl'.format(ii, extra_available_index)))


	## resume model from resume_epoch
	def resume_model(self, resume_epoch):
		for ii, _ in enumerate(self.model_list):
			self.model_list[ii].load_state_dict(
				torch.load(os.path.join(self.save_model_path, 'model-{}_{}.pkl'.format(ii, resume_epoch))))
		print('Time: {}, successfully resume model from {}'.format(time_now(), resume_epoch))

	## resume model from resume_epoch
	def resume_model_from_path(self, path, resume_epoch):
		for ii, _ in enumerate(self.model_list):
			self.model_list[ii].load_state_dict(
				torch.load(os.path.join(path, 'model-{}_{}.pkl'.format(ii, resume_epoch))))
		print('Time: {}, successfully resume model from {}'.format(time_now(), resume_epoch))

	## set model as train mode
	def set_train(self):
		for ii, _ in enumerate(self.model_list):
			self.model_list[ii] = self.model_list[ii].train()

	## set model as eval mode
	def set_eval(self):
		for ii, _ in enumerate(self.model_list):
			self.model_list[ii] = self.model_list[ii].eval()