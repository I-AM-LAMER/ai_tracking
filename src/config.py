# Configuration settings for the people tracking application

MODEL_PATH = "path/to/yolov8/model"  # Path to the pre-trained YOLOv8 model
DEEPSORT_MODEL_PATH = "path/to/deepsort/model"  # Path to the pre-trained DeepSORT model
API_HOST = "0.0.0.0"  # Host for the FastAPI application
API_PORT = 8000  # Port for the FastAPI application
WEBSOCKET_HOST = "0.0.0.0"  # Host for the WebSocket server
WEBSOCKET_PORT = 8765  # Port for the WebSocket server
VIDEO_SOURCE = 0  # Default video source (0 for webcam, or path to video file)
DEBUG_MODE = True  # Set to True for debug mode, False for production