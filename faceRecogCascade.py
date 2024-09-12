import cv2
import face_recognition

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load known face(s) and encode them
known_face_encodings = []
known_face_names = []

# Load known images and encode them
known_images = [
    {"name": "Dasuni", "path": "reference.jpeg"},
 #   {"name": "Person 2", "path": "path_to_image_of_person_2.jpg"}
]

for person in known_images:
    image = face_recognition.load_image_file(person['path'])
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)
    known_face_names.append(person['name'])

# Initialize the camera
video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame from the camera
    ret, frame = video_capture.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using the Haar Cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through all detected faces
    for (x, y, w, h) in faces:
        # Extract the face region of interest (ROI)
        face_roi = frame[y:y+h, x:x+w]

        # Resize the face ROI to match the face_recognition expected input size
        rgb_face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

        # Find encodings for the detected face
        face_encodings = face_recognition.face_encodings(rgb_face_roi)

        # Loop through each face encoding (usually only one face is detected in the ROI)
        for face_encoding in face_encodings:
            # Compare the detected face with known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = face_distances.argmin()

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the name of the detected face
            cv2.rectangle(frame, (x, y + h - 35), (x + w, y + h), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (x + 6, y + h - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
video_capture.release()
cv2.destroyAllWindows()
