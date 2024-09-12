import threading
import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt
import os

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_match = False
reference_img = cv2.imread("reference.jpeg")
lock = threading.Lock()

# Detector backend (DeepFace handles this automatically in verify)
detector_backend = 'mtcnn'

# Detect and align the reference face
try:
    result_ref = DeepFace.analyze(img_path=reference_img, detector_backend=detector_backend, actions=['age', 'gender', 'race', 'emotion'])
    # Assuming the face was detected, we proceed
    aligned_reference_img = result_ref['region']
    x, y, w, h = aligned_reference_img['x'], aligned_reference_img['y'], aligned_reference_img['w'], aligned_reference_img['h']
    reference_img = reference_img[y:y+h, x:x+w]
except Exception as e:
    print(f"Error: {e}")
    exit()

# Show the cropped reference image for debugging
plt.imshow(cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB))
plt.title('Cropped Reference Image')
plt.show()

def check_face(frame):
    global face_match
    try:
        # Compare the aligned faces
        result = DeepFace.verify(img1_path=frame, img2_path=reference_img, model_name='VGG-Face', detector_backend=detector_backend)
        with lock:
            face_match = result['verified']
            print(f"Distance: {result['distance']}")  # Debugging distance
    except Exception as e:
        with lock:
            face_match = False
            print(f"Error in face verification: {e}")

while True:
    ret, frame = cap.read()
    if ret:
        if counter % 30 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError:
                pass
        counter += 1

        with lock:
            if face_match:
                cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            else:
                cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
