import cv2
import numpy as np
import base64
import hashlib
import os
from flask import request, jsonify
from io import BytesIO
from PIL import Image
import base64

from app import app

# Load pre-trained model
net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'deploy.caffemodel')
CLASSES = [
    "background", "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

# Function to generate a unique filename
def generate_filename(file_path):
    file_name = os.path.basename(file_path)
    file_name, file_extension = os.path.splitext(file_name)
    hash_val = hashlib.md5(file_path.encode()).hexdigest()[:10]  # Generate MD5 hash of the file path
    return f"{file_name}_{hash_val}{file_extension}"

# Define folder path to save annotated images
ANNOTATIONS_FOLDER = '/Users/mohitjindal/Desktop/images/annotations/'
# Create ANNOTATIONS_FOLDER if it doesn't exist
if not os.path.exists(ANNOTATIONS_FOLDER):
    os.makedirs(ANNOTATIONS_FOLDER)

# Route to detect objects in uploaded image
@app.route('/detect', methods=['POST'])
def detect_objects():
    # Get file path from JSON payload
    data = request.get_json()
    file_path = data['file_path']

    # Load image from file path
    image = Image.open(file_path)


    # Convert image to RGB (3 channels)
    image = image.convert("RGB")
    
    np_image = np.array(image)
    (h, w) = np_image.shape[:2]

    # Preprocess image
    blob = cv2.dnn.blobFromImage(cv2.resize(np_image, (300, 300)), 0.007843, (300, 300), 127.5)

    # Pass the blob through the network and obtain the detections
    net.setInput(blob)
    detections = net.forward()

    # Initialize list to store annotations
    annotations = []

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        # Filter out weak detections
        if confidence > 0.8:
            # Extract class label
            class_id = int(detections[0, 0, i, 1])
            class_name = CLASSES[class_id]

            # Calculate bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Add annotation to the list
            annotation = {
                'class_name': class_name,
                'confidence_score': float(confidence),
                'bounding_box': {
                    'left': int(startX),
                    'top': int(startY),
                    'width': int(endX - startX),
                    'height': int(endY - startY)
                }
            }
            annotations.append(annotation)

            # Draw bounding box
            cv2.rectangle(np_image, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # Put text label
            label = f"{class_name}: {confidence * 100:.2f}%"
            y = startY - 15 if startY - 15 > 15 else startY + 15
            x = startX
            cv2.putText(np_image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

    # Convert image to base64 encoding
    # _, buffer = cv2.imencode('.jpg', np_image)
    # img_str = base64.b64encode(buffer)
            
    # Generate unique file name for annotated image
    # file_name = os.path.basename(file_path)
    annotated_file_name = generate_filename(file_path)
    annotated_file_path = os.path.join(ANNOTATIONS_FOLDER, annotated_file_name)

    # Save annotated image to specified folder
    cv2.imwrite(annotated_file_path, np_image)

    # Return the modified image along with annotations in the response
    response_data = {
        # 'image': img_str.decode('utf-8'),
        'annotated_image_path': annotated_file_path,
        'annotations': annotations
    }
    return jsonify(response_data)
