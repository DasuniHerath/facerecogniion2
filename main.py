import cv2
import face_recognition
from simple_facerec import SimpleFacerec
import pickle
import time
import math

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# Load Camera
camera_index = 0
cap = cv2.VideoCapture(camera_index)

fps = 0
fps_start_time = 0      #fps 2 method
fps_end_time = 0                #fps 2 method

if not cap.isOpened():
    print(f"Error: Camera with index {camera_index} could not be opened.")

# Load encodings from a file if it exists
try:
    with open('encodings.pkl', 'rb') as f:
        sfr.known_face_encodings, sfr.known_face_names = pickle.load(f)
except FileNotFoundError:
    print("No pre-saved encodings found. Using fresh encodings.")

frame_counter = 0  # Initialize frame counter

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image.")
        break
    
    frame_counter += 1  # Increment frame counter

    if frame_counter == 1:
        avg_fps = fps
    else:
        avg_fps = math.ceil((avg_fps * frame_counter + fps)/(frame_counter + 1))    

    #fps 2 method
    fps_end_time = time.time()
    time_diff = fps_end_time - fps_start_time
    fps = 1/(time_diff)
    fps_start_time = fps_end_time

    fps_text = "FPS: {:.2f}".format(fps)

    cv2.putText(frame, fps_text,(5,30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),1)
    cv2.putText(frame, "FPS avg:"+ str(avg_fps),(5,60),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),1)

    #fps 1 method
    # fps = int(cap.get(cv2.CAP_PROP_FPS))  # access FPS property
    # print("fps:", fps)  # print fps
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(frame, str(fps), (50, 50), font, 1, (0, 0, 255), 2)

    # Measure time before face recognition
    start_time = time.time()

    # Detect faces
    face_locations, face_names = sfr.detect_known_faces(frame)
    
    # Measure time after face recognition
    recognition_time = time.time() - start_time  # Time taken for face recognition

    #avg time
    avg_time =   recognition_time /frame_counter

    if face_names:  # If faces are recognized
        print(f"Faces recognized in frame {frame_counter}: {face_names}")
        print(f"Time to recognize faces: {recognition_time:.4f} seconds")
        print(f"avg time : {avg_time}  ")

    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC key to break
        break

# Save encodings to a file
with open('encodings.pkl', 'wb') as f:
    pickle.dump((sfr.known_face_encodings, sfr.known_face_names), f)

cap.release()
cv2.destroyAllWindows()
