import h5py
import numpy as np
import os


def run_matlab_evaluate_file(matlab, feature_dir, result_dir, mode, number_shot):

	settings = ' -nodesktop -nosplash -nojvm -r'

	matlab_workspace = os.path.join(os.getcwd(), 'tools/evaluation/sysu_mm01/matlab_files/')
	operator_1 = '''"addpath('{}')'''.format(matlab_workspace)
	operator_2 = '''evaluate('{}','{}','{}','{}',{});quit;"'''.format(matlab_workspace, feature_dir, result_dir, mode, number_shot)

	operator = ''' {};{}'''.format(operator_1, operator_2)
	command = matlab + settings + operator

	os.system(command)


def read_m_file_results(file_path):
	'''load result from mat file and return as list'''
	result = h5py.File(file_path)
	cmc_mean = result['performance']['cmc_mean'].value
	cmc_mean = np.resize(cmc_mean, [len(cmc_mean)])
	map_mean = result['performance']['map_mean'].value
	map_mean = np.resize(map_mean, [len(map_mean)])
	return cmc_mean, map_mean

