import os
import cv2
import face_recognition
import numpy as np
from tqdm import tqdm

class Face:
    def __init__(self, bounding_box, cropped_face, name, feature_vector):
        self.bounding_box = bounding_box
        self.cropped_face = cropped_face
        self.name = name
        self.feature_vector = feature_vector
        
def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def create_database(filenames, images):
    faces = []
    for filename, image in tqdm(zip(filenames, images), total=len(filenames)):
        loc = face_recognition.face_locations(image, model='hog')[0]
        vec = face_recognition.face_encodings(image, [loc], num_jitters=20)[0]
        
        top, right, bottom, left = loc
        
        cropped_face = image[top:bottom, left:right]
        
        face = Face(bounding_box=loc, cropped_face=cropped_face, name=filename.split('.')[0], feature_vector=vec)
        faces.append(face)
    
    return faces

def detect_faces(image_test, faces, threshold=0.6):
    locs_test = face_recognition.face_locations(image_test, model='hog')
    vecs_test = face_recognition.face_encodings(image_test, locs_test, num_jitters=1)
    
    for loc_test, vec_test in zip(locs_test, vecs_test):
        distances = []
        for face in faces:
            distance = face_recognition.face_distance([vec_test], face.feature_vector)
            distances.append(distance)
            
        if np.min(distances) > threshold:
            pred_name = 'unknown'
        else:
            pred_index = np.argmin(distances)
            pred_name = faces[pred_index].name
        
        image_test = draw_bounding_box(image_test, loc_test)
        image_test = draw_name(image_test, loc_test, pred_name)
    
    return image_test

def draw_bounding_box(image_test, loc_test):
    top, right, bottom, left = loc_test
    
    line_color = (0, 255, 0)
    line_thickness = 2
    
    cv2.rectangle(image_test, (left, top), (right, bottom), line_color, line_thickness)
    return image_test

def draw_name(image_test, loc_test, pred_name):
    top, right, bottom, left = loc_test 
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_color = (0, 0, 255)
    line_thickness = 3
    
    text_size, _ = cv2.getTextSize(pred_name, font, font_scale, line_thickness)
    
    bg_top_left = (left, top-text_size[1])
    bg_bottom_right = (left+text_size[0], top)
    cv2.rectangle(image_test, bg_top_left, bg_bottom_right, (0, 255, 0), -1)   

    cv2.putText(image_test, pred_name, (left, top), font, font_scale, font_color, line_thickness)
    
    return image_test

filenames = os.listdir('templates')
images = [load_image(f'templates/{filename}') for filename in filenames]
faces = create_database(filenames, images)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
	_, image, = cap.read()
	image = cv2.flip(image, flipCode=1)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	image = detect_faces(image, faces, threshold=0.6)

	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	cv2.imshow('image', image)