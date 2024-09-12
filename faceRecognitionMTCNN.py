from mtcnn import MTCNN
import cv2

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize MTCNN face detector
detector = MTCNN()

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces in the frame
    output = detector.detect_faces(frame)
    
    # Draw rectangles around detected faces
    for single_output in output:
        x, y, width, height = single_output['box']
        cv2.rectangle(frame, pt1=(x, y), pt2=(x + width, y + height), color=(255, 0, 0), thickness=3)

    # Display the resulting frame
    cv2.imshow('win', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
