########################################################################
############################  UTILS  ###################################
########################################################################

import os
import shutil
import numpy as np


# Delete all conent in the folder
def clear_directory(folder_path):
	for the_file in os.listdir(folder_path):
		file_path = os.path.join(folder_path, the_file)
		try:
			if os.path.isfile(file_path):
				os.unlink(file_path)
			elif os.path.isdir(file_path): 
				shutil.rmtree(file_path)
		except Exception as e:
				print(e)

# Create a new folder to store the different steps of algorithm
def create_new_folder(path_to_directory):
	try:
		os.makedirs(path_to_directory)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise

# Convert a PIL image to a csv one
def convert_PIL_to_CV(pil_image):
	open_cv_image = np.array(pil_image)
	# Convert RGB to BGR 
	open_cv_image = open_cv_image[:, :, ::-1].copy() 
	return open_cv_image

# convert image from pil to numpy array
def load_image_into_numpy_array(image):
	(im_width, im_height) = image.size
	return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

# prepare data to write as csv file
def output_data(text_boxes, text_words, image_width, image_height):
	output_data_values = []
	columns_name = ['image_width', 'image_height', 'box_xmin', 'box_xmax', 'box_ymin', 'box_ymax', 'predicted_word']

	for i in xrange(0,len(text_boxes)):
		xmin, ymin, xmax, ymax = text_boxes[i]
		word = text_words[i]
		values = (image_width,
			image_height,
			xmin,
			ymin,
			xmax,
			ymax,
			word
			)
		output_data_values.append(values)

	return columns_name, output_data_values