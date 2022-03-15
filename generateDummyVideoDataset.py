import os
import shutil
import argparse
import numpy as np
from PIL import Image

np.random.seed(42)


def generateDummyVideoDataset(opt):
	output_dir = opt.output_dir
	videos_nr = opt.videos_nr 
	frames_mean, frames_std = opt.frames_mean, opt.frames_std
	labels_mean, labels_std = opt.labels_mean, opt.labels_std

	frames2labels_array = np.zeros((2, videos_nr), dtype=int)
	frames2labels_array[0,:] = np.ceil(np.abs(np.random.normal(frames_mean, frames_std, videos_nr)))
	frames2labels_array[1,:] = np.ceil(np.abs(np.random.normal(labels_mean, labels_std, videos_nr)))

	dummy_img = Image.new('L', (1, 1))

	for video_id in range(videos_nr):
		video_path = output_dir + f"video_{str(video_id).zfill(5)}/"
		frames_path = video_path + '/images/'
		labels_path = video_path + '/labels/'
		os.makedirs(frames_path)
		os.makedirs(labels_path)

		for frame_id in range(frames2labels_array[0, video_id]):
			frame_name = f"video_{str(video_id).zfill(5)}_{str(frame_id).zfill(5)}.jpg"
			label_name = frame_name.replace('jpg', 'txt')
			frame_path = frames_path + frame_name
			label_path = labels_path + label_name

			dummy_img.save(frame_path)

			labels_nr = frames2labels_array[1, video_id]
			dummy_label_txt = open(label_path, 'w')
			[dummy_label_txt.write(f'{j}\n') for j in range(labels_nr)]


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--videos_nr', type=int, default=100, help='Number of videos')
	parser.add_argument('--frames_mean', type=float, default=50, help='Frames mean per video')
	parser.add_argument('--frames_std', type=float, default=30, help='Frames std per video')
	parser.add_argument('--labels_mean', type=float, default=20, help='Labels mean per image')
	parser.add_argument('--labels_std', type=float, default=30, help='Labels std per image')
	parser.add_argument('--output_dir', type=str, default='dummy_video_dataset/', help='Output path directory')
	opt = parser.parse_args()

	try:
		if(os.path.isdir(opt.output_dir)):
			shutil.rmtree(opt.output_dir)
		os.makedirs(opt.output_dir, exist_ok = True)

		generateDummyVideoDataset(opt)

		print(f"Directory '{opt.output_dir}' created successfully")

	except OSError as error:
		print(f"Directory '{opt.output_dir}' can not be created")
		print(error)

	