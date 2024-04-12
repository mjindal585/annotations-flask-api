import cv2
import numpy as np
import base64
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
9
# Route to detect objects in uploaded image
@app.route('/detect', methods=['POST'])
def detect_objects():
    # Get uploaded image
    file = request.files['image']
    image = Image.open(BytesIO(file.read()))

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
            cv2.putText(np_image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert image to base64 encoding
    _, buffer = cv2.imencode('.jpg', np_image)
    img_str = base64.b64encode(buffer)

    # Return the modified image along with annotations in the response
    response_data = {
        'image': img_str.decode('utf-8'),
        'annotations': annotations
    }
    return jsonify(response_data)
