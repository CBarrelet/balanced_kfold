import os
import shutil
import argparse
from glob import glob
from tqdm import tqdm

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt


def balancedKFolding(opt):
	n_k, iterations = opt.k, opt.ite

	video_paths = glob(opt.input_dir + '/*')
	video_paths.sort()

	video_frame_count = [len(glob(path+'/images/*')) for path in video_paths]
	label_paths_per_videos = [glob(path+'/labels/*') for path in video_paths]

	dataset = {i: path for i, path in enumerate(video_paths)}

	data_array = np.zeros((len(video_paths), 3), dtype=int)

	for i, frame_count in enumerate(video_frame_count):
		data_array[i,0] = i
		data_array[i,1] = frame_count
		for label_path in label_paths_per_videos[i]:
			data_array[i,2] += len(open(label_path, 'r').readlines())

	videos_nr = data_array.shape[0]
	frames_nr = np.sum(data_array[:,1])
	labels_nr = np.sum(data_array[:,2])
	frames_fold_nr = frames_nr/n_k
	labels_fold_nr = labels_nr/n_k

	print("Total videos:", videos_nr)
	print("Total frames:", np.sum(data_array[:,1]))
	print("Total labels:", np.sum(data_array[:,2]))

	print("Target images/fold:", frames_fold_nr)
	print("Target labels/fold:", labels_fold_nr)

	frames_fold_stds = np.zeros(iterations, dtype=float)
	labels_fold_stds = np.zeros(iterations, dtype=float)

	frames_per_folds = np.zeros(n_k, dtype=int)
	labels_per_folds = np.zeros(n_k, dtype=int)

	for i in tqdm(range(iterations)):
		seed = i
		new_data_array = np.array(data_array)

		np.random.seed(seed)
		np.random.shuffle(new_data_array)

		frames_per_folds *= 0
		labels_per_folds *= 0

		# Cumulative frames stacking per video
		frames_stack = np.zeros(videos_nr, dtype=int)
		frames_stack[0] = new_data_array[0,1]
		for j in range(1, frames_stack.shape[0]):
			frames_stack[j] = new_data_array[j,1] + frames_stack[j-1]

		min_id = 0
		min_frames_nr = 0
		for j in range(1, n_k+1):
			max_frames_nr = int(j*frames_fold_nr)

			mask = np.where((frames_stack >= min_frames_nr) & (frames_stack < max_frames_nr))
			max_id = mask[0][-1]

			frames_per_folds[j-1] = np.sum(new_data_array[min_id:max_id,1])
			labels_per_folds[j-1] = np.sum(new_data_array[min_id:max_id,2])
			min_id = max_id
			min_frames_nr = max_frames_nr

		frames_fold_stds[i] = np.std(frames_per_folds)
		labels_fold_stds[i] = np.std(labels_per_folds)

	stds = np.column_stack((frames_fold_stds, labels_fold_stds))
	args_min = [np.argmin((1-0.1*alpha)*stds[:,0] + 0.1*alpha*stds[:,1]) for alpha in range(11)]

	res = np.zeros((11,4))
	res[:,0] = np.arange(0, 1.1, 0.1)
	res[:,1] = np.transpose(args_min)
	res[:,2:] = stds[args_min]

	print("\nalpha\tseed\tstd_i\tstd_l")
	print(np.round(res,2))
	np.save(f'{opt.output_dir}/weighted_sum.npy', res)

	alpha = 4
	arg_min = args_min[4]

	best_seed = arg_min
	np.save(opt.output_dir + '/best_seed.npy', best_seed)

	print("\nImages std:", np.round(stds[arg_min,0],3))
	print("Labels std:", np.round(stds[arg_min,1],3))
	print("Best seed:", best_seed)

	fig = plt.figure()
	alphas = res[:, 0]
	stds_i = res[:, 2]
	stds_l = res[:, 3]
	stdi, = plt.plot(alphas, stds_i, label='Images/fold std')
	stdl, = plt.plot(alphas, stds_l, label='Labels/fold std')
	plt.legend(handles=[stdi, stdl])
	plt.title(f"Weighted sum for {n_k} folds images and labels std")
	plt.xlabel("Alpha")
	plt.ylabel("Std")
	plt.xticks(alphas)
	plt.savefig(f"{opt.output_dir}/weighted_sum.png", format='png')

	createFolds(opt, dataset, data_array)


def createFolds(opt, dataset, data_array):
	best_seed = np.load(opt.output_dir + '/best_seed.npy')
	n_k = opt.k

	videos_nr = data_array.shape[0]
	frames_nr = np.sum(data_array[:,1])
	frames_fold_nr = frames_nr/n_k

	# Get the best folds configuration
	np.random.seed(best_seed)
	np.random.shuffle(data_array)

	folds = []
	frames_stack = np.zeros(videos_nr, dtype=int)
	frames_stack[0] = data_array[0,1]
	for j in range(1, frames_stack.shape[0]):
		frames_stack[j] = data_array[j,1] + frames_stack[j-1]

	min_id = 0
	min_frames_nr = 0
	for j in range(1, n_k+1):
		max_frames_nr = int(j*frames_fold_nr)
		mask = np.where((frames_stack >= min_frames_nr) & (frames_stack < max_frames_nr))
		max_id = mask[0][-1]
		folds.append(data_array[min_id:max_id,0])
		min_id = max_id
		min_frames_nr = max_frames_nr

	# Create Yolo format images and labels directories
	images_dir = opt.output_dir + '/images/'
	labels_dir = opt.output_dir + '/labels/'

	os.makedirs(images_dir, exist_ok = True)
	os.makedirs(labels_dir, exist_ok = True)

	video_paths = glob(opt.intput_dir + '/*')
	video_paths.sort()

	for video_path in video_paths:
		frame_paths = glob(video_path + '/images/*')
		label_paths = glob(video_path + '/labels/*')
		frame_paths.sort()
		label_paths.sort()

		for frame_path, label_path in zip(frame_paths, label_paths):
			new_frame_name = os.path.basename(frame_path)
			new_label_name = new_frame_name.replace('jpg', 'txt')
			new_frame_path = os.path.abspath(images_dir) + '/' + new_frame_name
			new_label_path = os.path.abspath(labels_dir) + '/' + new_label_name

			shutil.copyfile(frame_path, new_frame_path)
			shutil.copyfile(label_path, new_label_path)

	print()
	for i in range(n_k):
		fold_path = opt.output_dir + f'/fold_{i+1}/'
		os.makedirs(fold_path, exist_ok = True)
		n_image_valid = 0
		n_image_train = 0
		n_label_valid = 0
		n_label_train = 0

		print(f"Fold {i+1} creation...")
		for j, fold in enumerate(tqdm(folds)):
			if(i == j):
				# Validation set
				valid_shapes_txt = open(fold_path + 'valid.shapes', 'w')
				valid_txt = open(fold_path + 'valid.txt', 'w')
				for k in range(fold.shape[0]):
					id_ = fold[k]
					video_path = dataset[id_]
					video_name = os.path.basename(video_path)
					frame_paths = glob(video_path + '/images/*')
					frame_paths.sort()
					for frame_path in frame_paths:
						n_image_valid += 1
						label_path = frame_path.replace('images', 'labels').replace('jpg', 'txt')
						n_label_valid += len(open(label_path, 'r').readlines())
						new_frame_name = os.path.basename(frame_path)
						new_frame_path = images_dir + new_frame_name
						new_frame_path = os.path.abspath(images_dir) + '/' + new_frame_name
						image = Image.open(frame_path)
						width, height = image.size
						valid_shapes_txt.write(f'{width} {height}\n')
						valid_txt.write(new_frame_path + '\n')
				valid_shapes_txt.close()
				valid_txt.close()

				# Substract validation fold
				train_folds = folds[:j] + folds[j+1:]

				# Train set
				train_shapes_txt = open(fold_path + 'train.shapes', 'w')
				train_txt = open(fold_path + 'train.txt', 'w')
				for fold in train_folds:
					for k in range(fold.shape[0]):
						id_ = fold[k]
						video_path = dataset[id_]
						video_name = os.path.basename(video_path)
						frame_paths = glob(video_path + '/images/*')
						frame_paths.sort()
						for frame_path in frame_paths:
							n_image_train += 1
							label_path = frame_path.replace('images', 'labels').replace('jpg', 'txt')
							n_label_train += len(open(label_path, 'r').readlines())
							new_frame_name = os.path.basename(frame_path)
							new_frame_path = images_dir + new_frame_name
							new_frame_path = os.path.abspath(images_dir) + '/' + new_frame_name
							image = Image.open(frame_path)
							width, height = image.size
							train_shapes_txt.write(f'{width} {height}\n')
							train_txt.write(new_frame_path + '\n')
				train_shapes_txt.close()
				train_txt.close()
		print(f"Fold {i+1} training set: {n_image_train} images / {n_label_train} labels")
		print(f"Fold {i+1} validation set: {n_image_valid} images / {n_label_valid} labels\n")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_dir', type=str, default='dummy_video_dataset/', help='Dummy raw UNO dataset directory path') # 'dummy_UNO_dataset' in the paper
	parser.add_argument('--output_dir', type=str, default='kfold_output', help='Output path directory')
	parser.add_argument('--delete_output_dir', action='store_true', help='Delete the output directory if aleardy exists')
	parser.add_argument('--k', type=int, default=5, help='Folds number') # 5 in the paper
	parser.add_argument('--ite', type=int, default=10000, help='Iterations number') # 100000000 in the paper
	opt = parser.parse_args()

	if(opt.delete_output_dir and os.path.isdir(opt.output_dir)):
		shutil.rmtree(opt.output_dir)
		print(f"Directory '{opt.output_dir}'' deleted")

	try:
		os.makedirs(opt.output_dir, exist_ok = False)
		balancedKFolding(opt)
		print(f"Directory '{opt.output_dir}' created successfully")

	except OSError as error:
		print(f"Directory '{opt.output_dir}' can not be created")
		assert len(os.listdir(opt.output_dir) ) == 0, f"Directory '{opt.output_dir}' is not empty.\nConsider deleting the existing output directory or removing its content"

	
