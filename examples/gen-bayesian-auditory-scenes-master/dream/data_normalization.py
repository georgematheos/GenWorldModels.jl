import numpy as np
from glob import glob
import sys

def get_mean_var(fns):

	begin = True
	for fn in fns:
		x = np.load(fn)
		x = x.flatten()
		if begin:
			x_list = x
			begin = False
		else:
			x_list = np.concatenate((x_list, x))

	return np.mean(x_list), np.var(x_list)

def main(dataset, chapter):

	data_folder = "/om2/user/mcusi/gen-bayesian-auditory-scenes/dreams/"
	chapter_folder = "{}_{:02d}/".format(dataset,chapter)
	f = data_folder + chapter_folder
	print("Dataset: {}".format(chapter_folder))
	n = 3000
	print("Using {} datapoints.".format(n))
	sys.stdout.flush()

	#Compute scene mean and variance
	fns = glob(f + "*scene*.npy")
	scene_mean, scene_variance = get_mean_var(fns[:n])
	print("Scene mean: {}".format(scene_mean))
	print("Scene variance: {}".format(scene_variance))
	sys.stdout.flush()

	#Compute element mean and variance 
	fns = glob(f + "*elem*.npy")
	element_mean, element_variance = get_mean_var(fns[:n])
	print("Element mean: {}".format(element_mean))
	print("Element variance: {}".format(element_variance))
	sys.stdout.flush()


if __name__ == '__main__':
	main(sys.argv[1],int(sys.argv[2]))