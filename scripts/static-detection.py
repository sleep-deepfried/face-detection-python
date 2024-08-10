import cv2
import boto3

# Initialize the Rekognition client
rekognition = boto3.client('rekognition')

# Load image from file
image_path = 'C:/Users/Earl/Desktop/face-detection/data/ricki2.jpg'
image = cv2.imread(image_path)

# Check if the image was loaded correctly
if image is None:
    print(f"Failed to load image from {image_path}")
else:
    # Resize the image (e.g., to 800x600 pixels)
    target_width = 800
    target_height = 800
    resized_image = cv2.resize(image, (target_width, target_height))

    # Convert resized image to bytes
    _, buffer = cv2.imencode('.jpg', resized_image)
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
        left = int(box['Left'] * resized_image.shape[1])
        top = int(box['Top'] * resized_image.shape[0])
        width = int(box['Width'] * resized_image.shape[1])
        height = int(box['Height'] * resized_image.shape[0])

        cv2.rectangle(resized_image, (left, top), (left + width, top + height), (0, 255, 0), 2)
        
        emotion = face_detail['Emotions'][0]['Type']
        gender = face_detail['Gender']['Value']
        age_range = f"{face_detail['AgeRange']['Low']} - {face_detail['AgeRange']['High']}"

        cv2.putText(resized_image, f"Emotion: {emotion}", (left, top - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(resized_image, f"Gender: {gender}", (left, top - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(resized_image, f"Age: {age_range}", (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Display the resulting image
    cv2.imshow('Facial Detection', resized_image)

    # Save the resulting image
    output_path = 'C:/Users/Earl/Desktop/face-detection/output/ricki_detected.jpg'
    cv2.imwrite(output_path, resized_image)

    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()
