import sys 
import dlib
from skimage import io 
import cv2 as cv 

def detect_and_display_faces_with_dlib():
  #Take the image file name from the command line 
  image = sys.argv[1]

  # Create a HOG face detector using the built-in dlib class 
  face_detector = dlib.get_frontal_face_detector()

  win = dlib.image_window()

  # load the image into an array 
  img = io.imread(image)

  #Run the HOG face detector on the image data.
  # The result will be the bounding boxes of the faces in our images.
  detected_faces = face_detector(img, 1)
  print(f"I FOUND {len(detected_faces)} faces in the file {image}")

  #Open a window on the desktop showing the image .
  win.set_image(img)
  # Loop through each face we found in the image.
  for i, face in enumerate(detected_faces):
    win.add_overlay(face)

  # wait until the user hit <enter> to close the window.
  dlib.hit_enter_to_continue()

def detect_faces():
    cap = cv.VideoCapture(0)
    face_detector = dlib.get_frontal_face_detector()
    while(True):
      _,frame = cap.read()
      detected_faces = face_detector(frame)
      for i, face in enumerate(detected_faces) :
        x1,y1,x2,y2 = face.left(), face.top(), face.right(), face.bottom()
        cv.rectangle(frame,(x1, y1),(x2, y2), (255,255,0), 2)
      cv.imshow("detected",frame)
      key = cv.waitKey(1) & 0xFF
      if key == ord('q'):
        break
if __name__ == "__main__":
    detect_faces()

