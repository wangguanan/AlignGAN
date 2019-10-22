import sys
sys.path.append('../')
import os

import torch

from tools import *



def train_a_step(config, base, loaders, current_step):

	# set train mode and learning rate decay
	base.set_train()
	base.lr_decay(current_step)

	gan_meter, ide_meter = AverageMeter(neglect_value=99.99), AverageMeter()

	for iteration in range(100):

		ide_titles, ide_values = train_feature_module_a_iter(config, base, loaders)
		gan_titles, gan_values = train_pixel_module_a_iter(config, base, loaders)

		gan_meter.update(gan_values, 1)
		ide_meter.update(ide_values, 1)

	return gan_titles, gan_meter.get_val_numpy(), ide_titles, ide_meter.get_val_numpy()


def train_pixel_module_a_iter(config, base, loaders):
	#########################################################################################################
	#                                                     Data                                              #
	#########################################################################################################
	## images
	real_rgb_images, rgb_pids, _, _ = loaders.rgb_train_iter_gan.next_one()
	real_ir_images, ir_pids, _, _ = loaders.ir_train_iter_gan.next_one()
	real_rgb_images, real_ir_images = real_rgb_images.to(base.device), real_ir_images.to(base.device)
	rgb_pids, ir_pids = rgb_pids.to(base.device), ir_pids.to(base.device)

	## fake images
	fake_rgb_images = base.G_ir2rgb(real_ir_images)
	fake_ir_images = base.G_rgb2ir(real_rgb_images)

	## features
	real_ir_features = base.encoder(base.process_images_4_encoder(real_ir_images, True, True))
	fake_ir_features = base.encoder(base.process_images_4_encoder(fake_ir_images, True, True))

	#########################################################################################################
	#                                                     Generator                                         #
	#########################################################################################################
	# gan loss
	gan_loss_rgb = base.criterion_gan_mse(base.D_rgb(fake_rgb_images), base.ones)
	gan_loss_ir = base.criterion_gan_mse(base.D_ir(fake_ir_images, real_ir_features.detach()), base.ones)

	gan_loss = (gan_loss_rgb + gan_loss_ir) / 2.0

	# cycle loss
	cycle_loss_rgb = base.criterion_gan_cycle(base.G_ir2rgb(fake_ir_images), real_rgb_images)
	cycle_loss_ir = base.criterion_gan_cycle(base.G_rgb2ir(fake_rgb_images), real_ir_images)
	cycle_loss = (cycle_loss_rgb + cycle_loss_ir) / 2.0

	# idnetity loss
	identity_loss_rgb = base.criterion_gan_identity(base.G_ir2rgb(real_rgb_images), real_rgb_images)
	identity_loss_ir = base.criterion_gan_identity(base.G_rgb2ir(real_ir_images), real_ir_images)
	identity_loss = (identity_loss_rgb + identity_loss_ir) / 2.0

	## task related loss
	if base.lambda_pixel_tri != 0 or base.lambda_pixel_cls != 0:

		_, _, _, real_ir_embedding_list = base.embeder(real_ir_features)
		_, _, fake_ir_logit_list, fake_ir_embedding_list = base.embeder(fake_ir_features)
		tri_loss_1 = base.compute_triplet_loss(fake_ir_embedding_list, real_ir_embedding_list,
		                                       real_ir_embedding_list, rgb_pids, ir_pids, ir_pids)
		tri_loss_2 = base.compute_triplet_loss(real_ir_embedding_list, fake_ir_embedding_list,
		                                       fake_ir_embedding_list, rgb_pids, ir_pids, ir_pids)

		_, cls_loss = base.compute_classification_loss(fake_ir_logit_list, rgb_pids)

	else:
		tri_loss_1 = torch.Tensor([0]).to(base.device)
		tri_loss_2 = torch.Tensor([0]).to(base.device)
		cls_loss = torch.Tensor([0]).to(base.device)

	## overall loss and optimize
	g_loss = gan_loss + 10 * cycle_loss + 5 * identity_loss + \
	         base.lambda_pixel_tri * (tri_loss_1 + tri_loss_2) + base.lambda_pixel_cls * cls_loss

	base.G_optimizer.zero_grad()
	g_loss.backward(retain_graph=True)
	base.G_optimizer.step()

	#########################################################################################################
	#                                                     Discriminator                                     #
	#########################################################################################################
	# discriminator rgb
	gan_loss_rgb_real = base.criterion_gan_mse(base.D_rgb(real_rgb_images), base.ones)
	gan_loss_rgb_fake = base.criterion_gan_mse(base.D_rgb(fake_rgb_images.detach()), base.zeros)
	gan_loss_rgb = (gan_loss_rgb_real + gan_loss_rgb_fake) / 2.0

	base.D_rgb_optimizer.zero_grad()
	gan_loss_rgb.backward()
	base.D_rgb_optimizer.step()

	# discriminator ir
	gan_loss_ir_real = base.criterion_gan_mse(base.D_ir(real_ir_images, real_ir_features.detach()), base.ones)

	wrong_real_ir_images, _ = base.generate_wrong_images(real_ir_images, ir_pids)
	gan_loss_ir_fake_1 = base.criterion_gan_mse(base.D_ir(wrong_real_ir_images, real_ir_features.detach()), base.zeros)
	gan_loss_ir_fake_2 = base.criterion_gan_mse(base.D_ir(real_ir_images, fake_ir_features.detach()), base.zeros)
	gan_loss_ir_fake_3 = base.criterion_gan_mse(base.D_ir(fake_ir_images.detach(), real_ir_features.detach()), base.zeros)
	gan_loss_ir_fake_4 = base.criterion_gan_mse(base.D_ir(fake_ir_images.detach(), fake_ir_features.detach()), base.zeros)

	gan_loss_ir_fake = (gan_loss_ir_fake_1 + gan_loss_ir_fake_2 + gan_loss_ir_fake_3 + gan_loss_ir_fake_4) / 4.0

	gan_loss_ir = (gan_loss_ir_real + gan_loss_ir_fake) / 2.0

	base.D_ir_optimizer.zero_grad()
	gan_loss_ir.backward()
	base.D_ir_optimizer.step()

	return ['G_GAN', 'CYC', 'IDENT', 'D_GAN', 'TRI', 'CLS'], \
	       torch.Tensor([gan_loss.data, cycle_loss.data, identity_loss.data, (gan_loss_rgb + gan_loss_ir).data,
	                     (tri_loss_1 + tri_loss_2).data, cls_loss.data])


def train_feature_module_a_iter(config, base, loaders):
	### load data
	real_rgb_images, rgb_pids, _, _ = loaders.rgb_train_iter_ide.next_one()
	real_ir_images, ir_pids, _, _ = loaders.ir_train_iter_ide.next_one()
	real_rgb_images, rgb_pids, real_ir_images, ir_pids = \
		real_rgb_images.to(base.device), rgb_pids.to(base.device), real_ir_images.to(base.device), ir_pids.to(base.device)

	### ide
	## forward
	with torch.no_grad():
		fake_ir_images = base.G_rgb2ir(real_rgb_images).detach()
	fake_ir_features = base.encoder(base.process_images_4_encoder(fake_ir_images, True, True))
	real_ir_features = base.encoder(base.process_images_4_encoder(real_ir_images, True, True))

	_, _, fake_ir_logits_list, fake_ir_embedding_list = base.embeder(fake_ir_features)
	_, _, real_ir_logits_list, real_ir_embedding_list = base.embeder(real_ir_features)

	## compute losses
	# classification loss
	fake_ir_acc, loss_fake_ir_cls = base.compute_classification_loss(fake_ir_logits_list, rgb_pids)
	real_ir_acc, loss_real_ir_cls = base.compute_classification_loss(real_ir_logits_list, ir_pids)
	loss_cls = loss_fake_ir_cls + loss_real_ir_cls
	# triplet loss
	loss_ir2rgb_triplet = base.compute_triplet_loss(fake_ir_embedding_list, real_ir_embedding_list,
	                                                real_ir_embedding_list, rgb_pids, ir_pids, ir_pids)
	loss_rgb2ir_triplet = base.compute_triplet_loss(real_ir_embedding_list, fake_ir_embedding_list,
	                                                fake_ir_embedding_list, ir_pids, rgb_pids, rgb_pids)
	loss_triplet = loss_ir2rgb_triplet + loss_rgb2ir_triplet

	# gan loss
	if base.lambda_feature_gan != 0.0:
		logits = base.D_ir(real_ir_images, fake_ir_features)
		loss_gan = base.criterion_gan_mse(logits, torch.ones_like(logits))
	else:
		loss_gan = torch.Tensor([0.0]).to(base.device)

	## overall loss
	loss = base.lambda_feature_cls * loss_cls + base.lambda_feature_triplet * loss_triplet + base.lambda_feature_gan * loss_gan

	## backward and optimize
	base.ide_optimizer.zero_grad()
	loss.backward()
	base.ide_optimizer.step()

	return ['fake_ir_acc', 'real_ir_acc', 'loss_fake_ir_cls', 'loss_real_ir_cls', 'loss_triplet', 'loss_gan'], \
	       torch.Tensor([fake_ir_acc[0], real_ir_acc[0], loss_fake_ir_cls.data, loss_real_ir_cls.data, loss_triplet.data,loss_gan.data])