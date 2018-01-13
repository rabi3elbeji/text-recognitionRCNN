############################################################
###############  TEXT DETECTION ALGORITHM  #################
############################################################

# imports 
import torch
import cv2
import sys
import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from torch.autograd import Variable
import utils
import TextDetection
from utils import image_utils
from utils import torch_utils
from utils import torch_dataset
from PIL import Image
import classifier_config.models.crnn as crnn

# Text Detection Global Variables
DETECTION_FROZEN_GRAPH_PATH = 'detector_config/frozen_inference_graph.pb'
DETECTION_CLASS_NUMBER = 1
PERCENT_PER_BOX_HEIGHT = 0.01  # Increase the height of selected box by 10% compared to the box height
PERCENT_PER_BOX_WIDTH = 0.03  # Increase the height of selected box by 8% compared to the box height


# Letter Classification Global Variables
CLASSIFICATION_MODEL_PATH 		= 'classifier_config/crnn.pth'
CLASSIFICATION_ALPHABET 		= '0123456789abcdefghijklmnopqrstuvwxyz'

# Test Global Variables
INPUT_IMAGES_PATH 	= 'inputs/'
OUTPUT_IMAGES_PATH = 'outputs/'
OUTPUT_IMAGES_PATH_STEPS = OUTPUT_IMAGES_PATH+"steps/"
OUTPUT_IMAGES_PATH_FILES = OUTPUT_IMAGES_PATH+"files/"
MIN_SCORE_DTECTION_THRESH = 0.5 # 50%
MAX_TEXT_BLOCK = 20 	# max words to be detected in the image

# Variables For Drawing
TEXT_FONT = cv2.FONT_HERSHEY_PLAIN
TEXT_FONT_SCALE = 2 
TEXT_THICKNESS = 2


def main():
	
	print ("Load images from the input dir ...")
	# Load images for test from the input dir
	input_images_path = os.listdir(INPUT_IMAGES_PATH)
	print ("Clear the output directory ...")
	# Clear the output directory
	image_utils.clear_directory(OUTPUT_IMAGES_PATH)
	image_utils.create_new_folder(OUTPUT_IMAGES_PATH_STEPS)
	image_utils.create_new_folder(OUTPUT_IMAGES_PATH_FILES)

	
	print("Init Text Detector Graph ...")
	# Init Detecor variabels
	detection_graph, detection_image_tensor, detection_boxes, detection_scores, detection_classes, num_detections = TextDetection.init_graph(DETECTION_FROZEN_GRAPH_PATH)
	
	print("Init Letter Classifier ...")
	# Init Classifier variabels and open tf session for classification
	model_classifier = crnn.CRNN(32, 1, 37, 256)
	# Use CUDA if exist
	if torch.cuda.is_available():
		model_classifier = model_classifier.cuda()
	# load the model weights 
	model_classifier.load_state_dict(torch.load(CLASSIFICATION_MODEL_PATH))
	# converter
	converter 	= torch_utils.strLabelConverter(CLASSIFICATION_ALPHABET)
	transformer = torch_dataset.resizeNormalize((100, 32))

	# open tf session for detection
	detection_session = tf.Session(graph=detection_graph)


	print("Start Processing!")
	start_processing = time.time()
	for input_image_name in input_images_path:
		word_text = []
		word_box  = []
		input_image_path = INPUT_IMAGES_PATH+str(input_image_name)
		# Get the base name from the path
		input_image_name = os.path.basename(input_image_name)
		print("Process image "+str(input_image_path)+" ... ")
		# prepare dirs
		output_image_dir_in_steps = OUTPUT_IMAGES_PATH_STEPS+"/"+input_image_name[:-4]
		image_utils.create_new_folder(output_image_dir_in_steps)
		# load image (in pil format)
		image_pil = Image.open(input_image_path)
		# get the size of the original image
		image_width, image_height = image_pil.size
		# convert pil to OpenCV format
		image_cv = image_utils.convert_PIL_to_CV(image_pil)
		# conserve a copy for drawing and show
		image_cv_for_draw = image_cv.copy()
		# the array based representation of the image will be used later in order to prepare the
		# result image with boxes and labels on it.
		image_np = image_utils.load_image_into_numpy_array(image_pil)
		# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
		image_np_expanded = np.expand_dims(image_np, axis=0)
		# Predict (Detect) text box
		detect_boxes, detect_classes, detect_scores = TextDetection.detect_text(detection_session, 
			detection_boxes, 
			detection_scores, 
			detection_classes, 
			num_detections, 
			detection_image_tensor, 
			image_np_expanded)

		print("Start detection boxes in image "+str(input_image_path)+" ... ")
		# variable to conctenate predicted letters 
		
		# loop the detected boxes
		for i in range(min(MAX_TEXT_BLOCK, detect_boxes.shape[0])):
			# test if score is heigher or equal to the pre-defined threshold
			if detect_scores is None or detect_scores[i] > MIN_SCORE_DTECTION_THRESH:

				word_name = 'word'+str(i+1)
			
				# Get word's box coord
				ymin, xmin, ymax, xmax = tuple(detect_boxes[i].tolist())
				word_xmin, word_xmax, word_ymin, word_ymax =  (int(xmin * image_width), int(xmax * image_width), int(ymin * image_height), int(ymax * image_height))
				
				# save the cordination for print
				saved_word_box = word_xmin, word_xmax, word_ymin, word_ymax
				word_box.append(saved_word_box)
				# Extract the word's ROI for the next  process
				word_roi = image_cv[word_ymin:word_ymax, word_xmin:word_xmax]
				# save the word ROI
				cv2.imwrite(output_image_dir_in_steps+"/"+word_name+".jpg", word_roi)

				####################################################################################
				######################## Start letter detection + classification ###################
				####################################################################################
				print("Start Letter Detection + classification in "+word_name+" ..")
				# Start RCNN classification
				word_roi = cv2.cvtColor(word_roi,cv2.COLOR_BGR2RGB)
				word_roi_pil = Image.fromarray(word_roi)

				word_roi_pil = word_roi_pil.convert('L')
				word_roi_pil = transformer(word_roi_pil)

				if torch.cuda.is_available():
					word_roi_pil = word_roi_pil.cuda()

				word_roi_pil = word_roi_pil.view(1, *word_roi_pil.size())
				word_roi_pil = Variable(word_roi_pil)

				# prepare torch to eval 
				model_classifier.eval()
				preds = model_classifier(word_roi_pil)
				_, preds = preds.max(2)
				preds = preds.transpose(1, 0).contiguous().view(-1)

				preds_size = Variable(torch.IntTensor([preds.size(0)]))
				raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
				sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
				# save predicted word
				word_text.append(sim_pred)	

				print('%-20s => %-20s' % (raw_pred, sim_pred))
	
				print("End Letter Detection + classification in "+word_name+" ..")

				# Draw informations on the original image
				label_top_left = (word_xmin , word_ymin - 30)
				label_bottom_right = (word_xmax , word_ymin)					
				cv2.rectangle(image_cv_for_draw, label_top_left, label_bottom_right, (0,255,0), -1)
				cv2.putText(image_cv_for_draw, sim_pred, (word_xmin+5, word_ymin-3), TEXT_FONT, TEXT_FONT_SCALE, (0, 0, 0), TEXT_THICKNESS)
				cv2.rectangle(image_cv_for_draw,(word_xmin, word_ymin),(word_xmax, word_ymax),(0,255,0),2)


		print("End detection boxes in image "+str(input_image_path))
		cv2.imwrite(OUTPUT_IMAGES_PATH+input_image_name, image_cv_for_draw)

		columns_name, file_data = image_utils.output_data(word_box, word_text, image_width, image_height)
		csv_df = pd.DataFrame(file_data, columns=columns_name)
		csv_df.to_csv(OUTPUT_IMAGES_PATH_FILES+input_image_name[:-3]+'.csv', index=None)

	end_processing = time.time()
	print("End Processing!, elapsed time = "+str(end_processing - start_processing)+" s")
	 


if __name__ == "__main__":
	main()
