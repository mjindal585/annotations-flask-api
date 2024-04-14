# annotations-flask-api
This Flask API provides an endpoint for detecting objects in images using an AI model. It returns annotated images with bounding boxes and object names.

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd object-detection-api
```
2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Usage

1. Run the Flask server:

```bash
python run.py
```

2. Make a POST request to the /detect endpoint with an image file:

```bash
curl -X POST -F "image=@/path/to/image.jpg" http://localhost:8000/detect
```

Note: Replace /path/to/image.jpg with the path to your image file.

The API will return a JSON response containing the annotated image file path and object annotations

```plaintext
{
    "image_path": "/path/to/annotated/image.jpg",
    "annotations": [
        {
            "class_name": "car",
            "confidence_score": 0.92,
            "bounding_box": {
                "left": 100,
                "top": 50,
                "width": 200,
                "height": 150
            }
        },
        {
            "class_name": "person",
            "confidence_score": 0.85,
            "bounding_box": {
                "left": 300,
                "top": 200,
                "width": 150,
                "height": 300
            }
        }
    ]
}
```

### Directory Structure

```plaintext
object-detection-api/
├── app/
│   ├── __init__.py
│   ├── detection.py
├── requirements.txt
├── run.py
└── deploy.prototxt.txt
└── deploy.caffemodel
```

app/: Contains Flask application files.
requirements.txt: List of Python dependencies.
run.py: Entry point for running the Flask server.
deploy.prototxt.txt and deploy.caffemodel: Pre-trained model files for object detection.

### Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

### License
MIT
