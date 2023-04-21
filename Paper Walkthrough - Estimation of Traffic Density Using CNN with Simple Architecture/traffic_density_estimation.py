# Load modules.
import cv2
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Check whether GPU is available.
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Load the trained CNN model.
model = load_model('cnn_block_prediction.h5', compile=False)
print(model.summary())

# Loading the Junction 1 video.
cap = cv2.VideoCapture('Junction.avi')

# Every single block is perceived as an object with the following attributes.
class Square_Region():
	def __init__(self, topleft_x, topleft_y, bottomright_x, bottomright_y):
		self.topleft_x         = topleft_x
		self.topleft_y         = topleft_y
		self.bottomright_x     = bottomright_x
		self.bottomright_y     = bottomright_y
		
		# Topleft and bottomright corner coordinate of each block.
		self.topleft           = (self.topleft_x, self.topleft_y)
		self.bottomright       = (self.bottomright_x, self.bottomright_y)

		# Later we will put the block in this attribute.
		self.square_image	   = None

# Topleft and bottomright corner coordinates.
topleft_xs = [143, 143+20, 143+40, 
     158, 158+20, 158+40, 
     168, 168+20, 168+40, 168+60]

topleft_ys = [92, 92, 92, 
    92+20, 92+20, 92+20, 
    92+40, 92+40, 92+40, 92+40]

bottomright_xs = [163, 163+20, 163+40, 
     178, 178+20, 178+40, 
     188, 188+20, 188+40, 188+60]

bottomright_ys = [112, 112, 112, 
      112+20, 112+20, 112+20, 
      112+40, 112+40, 112+40, 112+40]


# The number of blocks to be initialized.
N_SQUARES = 10
square_regions = [None] * N_SQUARES

# Initializing Square_Region (block) objects.
for i in range(N_SQUARES):
	square_regions[i] = Square_Region(topleft_xs[i], topleft_ys[i], bottomright_xs[i], bottomright_ys[i])

# Helper variables for calculating processing time.
fps_max = 0
fps_avg_array = []
frame_count = 0
pTime = 0

# Reading every single frame in the video one by one.
while True:
	bois = []
	_, image = cap.read()
	frame_count += 1

	# Assigning the cropped images to square_image attribute.
	for i in range(N_SQUARES):
		square_region = square_regions[i]
		square_region.square_image = image[square_region.topleft_y:square_region.topleft_y+20, square_region.topleft_x:square_region.topleft_x+20]
		bois.append(square_region.square_image)
	bois = np.asarray(bois)

	# Predicting block occupancy.
	predictions = np.round(model.predict(bois))
	no_occupied = predictions.flatten().tolist().count(1)

	# Traffic density calculation.
	density = no_occupied/len(predictions)

	# For each block, if it is predicted as unoccupied then the block is colored green, else red.
	for i in range(N_SQUARES):
		square_region = square_regions[i]
		if predictions[i] == [0]:
			image = cv2.rectangle(image, (square_region.topleft), (square_region.bottomright), (0,255,0), 1)
		elif predictions[i] == [1]:
			image = cv2.rectangle(image, (square_region.topleft), (square_region.bottomright), (0,0,255), 1)
	
	# Helper variables for FPS calculation.
	cTime = time.time()
	fps = 1/(cTime-pTime)
	pTime = cTime

	if fps > fps_max:
		fps_max = np.round(fps, 2)
	fps_avg_array.append(fps)
	
	# Printing out FPS and traffic density information on the frame.
	print('fps_max: {}\tframe_count: {}'.format(fps_max, frame_count))
	cv2.putText(image, 'FPS: {}'.format(np.round(fps,1)), (10,255), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
	cv2.putText(image, 'Density: {}'.format(density), (10,270), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

	# Show the processed frame alongside with the prediction results.
	cv2.imshow('image', image)
	
	# Press 'q' to terminate the video.
	if cv2.waitKey(1) & 0XFF==ord('q'):
		break

# Calculating average FPS after the video is terminated.
fps_avg = np.mean(fps_avg_array)
print('fps_avg: {}'.format(np.round(fps_avg,2)))