import cv2
import boto3
import time

# Initialize the Rekognition client
rekognition = boto3.client('rekognition')

# Initialize webcam capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Convert frame to bytes
    _, buffer = cv2.imencode('.jpg', frame)
    image_bytes = buffer.tobytes()

    # Call Rekognition to detect faces
    response = rekognition.detect_faces(
        Image={'Bytes': image_bytes},
        Attributes=['ALL']
    )
    
    # Parse the response
    for face_detail in response['FaceDetails']:
        # Draw bounding box and display details
        box = face_detail['BoundingBox']
        left = int(box['Left'] * frame.shape[1])
        top = int(box['Top'] * frame.shape[0])
        width = int(box['Width'] * frame.shape[1])
        height = int(box['Height'] * frame.shape[0])

        cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 2)
        
        emotion = face_detail['Emotions'][0]['Type']
        gender = face_detail['Gender']['Value']
        age_range = f"{face_detail['AgeRange']['Low']} - {face_detail['AgeRange']['High']}"

        cv2.putText(frame, f"Emotion: {emotion}", (left, top - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Gender: {gender}", (left, top - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Age: {age_range}", (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
