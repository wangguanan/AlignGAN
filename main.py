import os
import argparse

import torchvision.transforms as transforms

from core import *
from tools import *



def main(config):

	# environments
	make_dirs(config.save_path)
	make_dirs(os.path.join(config.save_path, 'logs/'))
	make_dirs(os.path.join(config.save_path, 'model/'))
	make_dirs(os.path.join(config.save_path, 'features/'))
	make_dirs(os.path.join(config.save_path, 'results/'))
	make_dirs(os.path.join(config.save_path, 'images/'))


	# loaders
	transform_train = transforms.Compose([
		transforms.Resize([config.image_size, config.image_size], interpolation=3),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
	transform_test = transforms.Compose([
		transforms.Resize([config.image_size, config.image_size], interpolation=3),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
	loaders = Loaders(config, transform_train, transform_test)


	# base
	base = Base(config, loaders)


	# logger
	logger = Logger(os.path.join(os.path.join(config.save_path, 'logs/'), 'logging.txt'))
	logger(config)


	if config.mode == 'train':

		# automatically resume model from the latest one
		start_train_epoch = 0
		if True:
			root, _, files = os_walk(base.save_model_path)
			if len(files) > 0:
				# get indexes of saved models
				indexes = []
				for file in files:
					indexes.append(int(file.replace('.pkl', '').split('_')[-1]))

				# remove the bad-case and get available indexes
				model_num = len(base.model_list)
				available_indexes = copy.deepcopy(indexes)
				for element in indexes:
					if indexes.count(element) < model_num:
						available_indexes.remove(element)

				available_indexes = sorted(list(set(available_indexes)), reverse=True)
				unavailable_indexes = list(set(indexes).difference(set(available_indexes)))

				if len(available_indexes) > 0:  # resume model from the latest model
					base.resume_model(available_indexes[0])
					start_train_epoch = available_indexes[0]
					logger('Time: {}, automatically resume training from the latest step (model {})'.
						   format(time_now(), available_indexes[0]))
				else:  #
					logger('Time: {}, there are no available models')

		# train loop
		for current_step in range(start_train_epoch, config.warmup_feature_module_steps + config.warmup_pixel_module_steps + config.joint_training_steps):

			# save model every step. extra models will be automatically deleted for saving storage
			base.save_model(current_step)

			# evaluate reid
			if current_step in config.evaluate_reid_steps:
				logger('**********' * 10 + 'evaluate' + '**********' * 10)
				results = test(config, base, loaders, True)
				for key in list(results.keys()):
					logger('Time: {}, {}, {}'.format(time_now(), key, results[key]))
				logger('')

			# save fake infrared images
			if current_step >= config.warmup_pixel_module_steps:
				save_images(base, current_step)


			# warm up the feature alignment module
			if current_step < config.warmup_feature_module_steps:
				logger('**********' * 10 + 'warmup the feature module' + '**********' * 10)
				results_names, resluts_values = warmup_feature_module_a_step(config, base, loaders)
				logger('Time: {};  Step: {};  {}'.format(time_now(), current_step, analyze_names_and_meter(results_names, resluts_values)))
				logger('')

			# warm up the pixel alignment module
			elif current_step < config.warmup_feature_module_steps + config.warmup_pixel_module_steps:
				# save fake images
				save_images(base, current_step)
				# warm up
				logger('**********' * 10 + 'warmup the pixel module' + '**********' * 10)
				results_names, resluts_values = warmup_pixel_module_a_step(config, base, loaders)
				logger('Time: {};  Step: {};  {}'.format(time_now(), current_step, analyze_names_and_meter(results_names, resluts_values)))
				logger('')

			# jointly train the whole model
			else:
				logger('**********'*10 + 'train' + '**********'*10 )
				gan_titles, gan_values, ide_titles, ide_values = train_a_step(config, base, loaders, current_step)
				logger('Time: {};  Step: {};  {}'.format(time_now(), current_step, analyze_names_and_meter(gan_titles, gan_values)))
				logger('Time: {};  Step: {};  {}'.format(time_now(), current_step, analyze_names_and_meter(ide_titles, ide_values)))
				logger('')

		logger('**********' * 10 + 'final test' + '**********' * 10)
		results = test(config, base, loaders, False)
		for key in list(results.keys()):
			logger('Time: {}, {}, {}'.format(time_now(), key, results[key]))
		logger('')


	elif config.mode == 'test':

		base.resume_model_from_path(config.pretrained_model_path, config.pretrained_model_index)
		logger('**********' * 10 + 'test with pre-trained model' + '**********' * 10)
		results = test(config, base, loaders, False)
		for key in list(results.keys()):
			logger('Time: {}, {}, {}'.format(time_now(), key, results[key]))
		logger('')




if __name__ == '__main__':


	# Configurations
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', type=str, default='train')

	# output configuration
	parser.add_argument('--save_path', type=str, default='out/base/', help='path to save models, logs, images')
	# dataset configuration
	parser.add_argument('--dataset_path', type=str, default='SYSU-MM01/')
	parser.add_argument('--p_gan', type=int, default=4, help='person numbers for pixel alignment module')
	parser.add_argument('--k_gan', type=int, default=4, help='images numbers of a person for pixel alignment module')
	parser.add_argument('--p_ide', type=int, default=18, help='person numbers for feature alignment module')
	parser.add_argument('--k_ide', type=int, default=4, help='image numbers of a person for feature alignment module')
	parser.add_argument('--class_num', type=int, default=395, help='identity numbers in training set')
	parser.add_argument('--image_size', type=int, default=128, help='image size for pixel alignment module,. in feature alignment module, images will be automatically reshaped to 384*192')

	# restore configuration, used for debug, please don't change them
	parser.add_argument('--G_rgb2ir_restore_path', type=str, default='None')
	parser.add_argument('--G_ir2rgb_restore_path', type=str, default='None')
	parser.add_argument('--D_ir_restore_path', type=str, default='None')
	parser.add_argument('--D_rgb_restore_path', type=str, default='None')
	parser.add_argument('--encoder_restore_path', type=str, default='None')
	parser.add_argument('--embeder_restore_path', type=str, default='None')

	# [pixel align part] criterion configuration
	parser.add_argument('--lambda_pixel_tri', type=float, default=0.1)
	parser.add_argument('--lambda_pixel_cls', type=float, default=0.1)

	# [feature align part] criterion configuration
	parser.add_argument('--lambda_feature_cls', type=float, default=1.0)
	parser.add_argument('--lambda_feature_triplet', type=float, default=1.0)
	parser.add_argument('--lambda_feature_gan', type=float, default=0.1)
	parser.add_argument('--margin', type=float, default=1.0, help='margin for triplet loss')
	parser.add_argument('--soft_bh', type=float, default=[0, 0], help='parameters of triplet loss with batch hard')

	# training configuration
	parser.add_argument('--base_pixel_learning_rate', type=float, default=0.0002, help='learning rate for pixel alignment module')
	parser.add_argument('--base_feature_ide_learning_rate', type=float, default=0.2, help='learning rate for feature alignment module')

	# training configuration
	parser.add_argument('--warmup_feature_module_steps', type=int, default=50)
	parser.add_argument('--warmup_pixel_module_steps', type=int, default=100)
	parser.add_argument('--joint_training_steps', type=int, default=101)
	parser.add_argument('--milestones', nargs='+', type=int, default=[50])
	parser.add_argument('--save_model_steps', nargs='+', type=int, default=[100])

	# evaluate configuration
	parser.add_argument('--max_save_model_num', type=int, default=2, help='0 for max num is infinit, extra models will be automatically deleted for saving storage')
	parser.add_argument('--modes', type=str, nargs='+', default=['all', 'indoor'], help='')
	parser.add_argument('--number_shots', type=str, nargs='+', default=['single', 'multi'], help='')
	parser.add_argument('--matlab', type=str, default='none', help='in default, we use python evaluation code. additionally, we also support matlab evaluation version. please see code for more details')
	parser.add_argument('--evaluate_reid_steps', nargs='+', type=int, default=[100])

	# test configuration
	parser.add_argument('--pretrained_model_path', type=str, default='', help='please download the pretrained model at first, and then set path')
	parser.add_argument('--pretrained_model_index', type=int, default=None, help='')


	# parse
	config = parser.parse_args()
	config.milestones = list(np.array(config.milestones) + config.warmup_feature_module_steps + config.warmup_pixel_module_steps)
	config.save_model_steps = list(np.array(config.save_model_steps) + config.warmup_feature_module_steps + config.warmup_pixel_module_steps)
	config.evaluate_reid_steps = list(np.array(config.evaluate_reid_steps) + config.warmup_feature_module_steps + config.warmup_pixel_module_steps)

	main(config)
