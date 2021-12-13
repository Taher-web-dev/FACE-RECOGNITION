import sys 
import dlib
from skimage import io 
import cv2 as cv 
import numpy as np 
import openface.openface.align_dlib as openface
#import openface

predictor_model = 'pretained_model/shape_predictor_68_face_landmarks.dat'

def detect_and_display_faces_with_dlib():
  #Take the image file name from the command line 
  image = sys.argv[1]
  # Create a HOG face detector using the built-in dlib class 
  face_detector = dlib.get_frontal_face_detector()
  # pretained model for landmarks.
  face_pose_predictor = dlib.shape_predictor(predictor_model)
  # pretained model for image aligner.
  face_aligner = openface.AlignDlib(predictor_model)
  
  win = dlib.image_window()

  # load the image into an array 
  img = io.imread(image)

  #Run the HOG face detector on the image data.
  # The result will be the bounding boxes of the faces in our images.
  detected_faces = face_detector(img, 1)
  print(f"I FOUND {len(detected_faces)} faces in the file {image}")

  #Open a window on the desktop showing the image .
  #win.set_image(img)
  # Loop through each face we found in the image.
  for i, face in enumerate(detected_faces):
    #win.add_overlay(face)
    pose_landmark = face_pose_predictor(img, face)
    aligned_face = face_aligner.align(534, img, face, landmarkIndices= openface.AlignDlib.OUTER_EYES_AND_NOSE )
    win.set_image(aligned_face)

  # wait until the user hit <enter> to close the window.
  dlib.hit_enter_to_continue()

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

def detect_faces():
    cap = cv.VideoCapture(0)
    face_detector = dlib.get_frontal_face_detector()
    face_pose_predictor = dlib.shape_predictor(predictor_model)
    while(True):
      _,frame = cap.read()
      detected_faces = face_detector(frame)
      for i, face in enumerate(detected_faces) :
        x1,y1,x2,y2 = face.left(), face.top(), face.right(), face.bottom()
        cv.rectangle(frame,(x1, y1),(x2, y2), (255,255,0), 2)
        pose_landmark = face_pose_predictor(frame, face)
        shapes = shape_to_np(pose_landmark)
        for (x,y) in  shapes :
          cv.circle(frame,(x,y), 1, (255, 0, 0), -1)
      cv.imshow("detected",frame)
      key = cv.waitKey(1) & 0xFF
      if key == ord('q'):
        break
if __name__ == "__main__":
  detect_and_display_faces_with_dlib()
  #detect_faces()

