########################################################################
###################  THIS CLASS FOR Text DETECTION  ####################
########################################################################

import tensorflow as tf
import numpy as np
from utils import label_map_util


# Load a (frozen) Tensorflow model into memory.
def init_graph(path_to_frozen_graph):
	detection_graph = tf.Graph()
	with detection_graph.as_default():
		od_graph_def = tf.GraphDef()
		with tf.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')

	# Definite input and output Tensors for detection_graph
	image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
	# Each box represents a part of the image where a particular object was detected.
	detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
	# Each score represent how level of confidence for each of the objects.
	# Score is shown on the result image, together with the class label.
	detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
	detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
	num_detections = detection_graph.get_tensor_by_name('num_detections:0')
	return detection_graph, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections


# Loading label map
def load_labels(path_to_labels_file, num_classes):
	label_map = label_map_util.load_labelmap(path_to_labels_file)
	categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
	category_index = label_map_util.create_category_index(categories)
	return category_index

# Detect text box
def detect_text(detect_sess, detection_boxes, detection_scores, detection_classes, num_detections, image_tensor, image_np_expanded):
	(boxes, scores, classes, num) = detect_sess.run([detection_boxes, 
		detection_scores, 
		detection_classes, 
		num_detections],
		feed_dict={image_tensor: image_np_expanded})
	boxes = np.squeeze(boxes)
	classes = np.squeeze(classes).astype(np.int32)
	scores = np.squeeze(scores)

	return boxes, classes, scores

# Make the detected box a little big than his current size
def resize_box(xmin, xmax, ymin, ymax, image_height, image_width, percent_per_height, percent_per_width):
	# box height, width
	box_width  = xmax - xmin
	box_height = ymax - ymin
	# calculate new coordination
	new_xmin = xmin - int(box_width * percent_per_width)
	new_xmax = xmax + int(box_width * percent_per_width)
	new_ymin = ymin - int(box_height * percent_per_height) 
	new_ymax = ymax + int(box_height * percent_per_height)

	# Check image size limits
	if new_xmin < 0:
		new_xmin = 0
	if new_ymin < 0:
		new_ymin = 0
	if new_xmax > image_width:
		new_xmax = image_width
	if new_ymax > image_height:
		new_ymax = image_height

	return new_xmin, new_xmax, new_ymin, new_ymax