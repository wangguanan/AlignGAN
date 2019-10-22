import numpy as np
import scipy.io as sio
import os

import torch
from torchvision.utils import save_image

from tools import *



def test(config, base, loaders, brief):

	compute_and_save_features(base, loaders)
	results = evalutate(config, base, brief)
	return results


def evalutate(config, base, brief=False):

	results = {}
	for mode in config.modes:
		print(mode)
		for number_shot in config.number_shots:
			print(number_shot)
			cmc, map = evaluate_sysymm01(base.save_features_path, mode, number_shot)
			results['{},{}'.format(mode, number_shot)] = [cmc, map]
			if brief: break
		if brief: break

	return results


def compute_and_save_features(base, loaders):

	def compute_features(images):
		images_f = fliplr(images)
		images = images.to(base.device)
		images_f = images_f.to(base.device)
		features = base.encoder(base.process_images_4_encoder(images, True, True))
		features_f = base.encoder(base.process_images_4_encoder(images_f, True, True))
		features, _, _, _ = base.embeder(features)
		features_f, _, _, _ = base.embeder(features_f)
		features = features + features_f
		if base.part_num == 1:
			features = torch.unsqueeze(features, -1)
		return features

	def normalize_and_resize_feature(features):
		# normlize
		norm = torch.norm(features, dim=1, keepdim=True)
		features = features / norm.repeat([1, features.size(1), 1])
		# resize
		features = features.view(features.size(0), -1)
		return features

	class XX:
		def __init__(self):
			self.val = {}
		def update(self, key, value):
			if key not in self.val.keys():
				self.val[key] = value
			else:
				self.val[key] = np.concatenate([self.val[key], value], axis=0)
		def get_val(self, key):
			if key in self.val.keys():
				return self.val[key]
			else:
				return np.array([[]])


	print('Time:{}.  Start to compute features'.format(time_now()))
	# compute features
	# base._resume_model(test_step)
	base.set_eval()
	features_meter, pids_meter, cids_meter = CatMeter(), CatMeter(), CatMeter()

	with torch.no_grad():
		for i, data in enumerate(loaders.rgb_all_loader):
			# load data
			images, pids, cids, _ = data
			images = base.G_rgb2ir(images.to(base.device)).data.cpu()
			# forward
			features = compute_features(images)
			# meter
			features_meter.update(features.data)
			pids_meter.update(pids.data)
			cids_meter.update(cids.data)

		for i, data in enumerate(loaders.ir_all_loader):
			# load data
			images, pids, cids, _ = data
			# forward
			features = compute_features(images)
			# meter
			features_meter.update(features.data)
			pids_meter.update(pids.data)
			cids_meter.update(cids.data)

	print('Time:{}.  Start to normalize features.'.format(time_now()))
	# normalize features
	features = features_meter.get_val()
	features = normalize_and_resize_feature(features)
	features = features.data.cpu().numpy()
	pids = pids_meter.get_val_numpy()
	cids = cids_meter.get_val_numpy()

	print('Time: {}.  Note: Start to save features as .mat file'.format(time_now()))
	# save features as .mat file
	results = {1: XX(), 2: XX(), 3: XX(), 4: XX(), 5: XX(), 6: XX()}
	for i in range(features.shape[0]):
		feature = features[i, :]
		feature = np.resize(feature, [1, feature.shape[0]])
		cid, pid = cids[i], pids[i]
		results[cid].update(pid, feature)

	pid_num_of_cids = [333, 333, 533, 533, 533, 333]
	cids = [1, 2, 3, 4, 5, 6]
	for cid in cids:
		a_result = results[cid]
		xx = []
		for pid in range(1, 1+ pid_num_of_cids[cid - 1]):
			xx.append([a_result.get_val(pid).astype(np.double)])
		xx = np.array(xx)
		sio.savemat(os.path.join(base.save_features_path, 'feature_cam{}.mat'.format(cid)), {'feature': xx})



def save_images(base, current_step):

	#base.set_eval()
	with torch.no_grad():
		fixed_fake_ir_images = base.G_rgb2ir(base.fixed_real_rgb_images).detach()
		xxxx = torch.cat([base.fixed_real_rgb_images, fixed_fake_ir_images, base.fixed_real_ir_images], dim=0)
		save_image((xxxx.data.cpu() + 1.0) / 2.0,
		           os.path.join(base.save_images_path, 'image_{}.jpg'.format(current_step)), nrow=base.fixed_real_rgb_images.size(0), padding=0)