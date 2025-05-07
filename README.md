# People Tracking Project

This project implements a real-time people tracking system using YOLOv8 for detection and DeepSORT for tracking. The application provides an API for tracking video frames and a WebSocket server for real-time visualization of tracking data.

## Project Structure

```
ai_tracking
├── src
│   ├── api
│   │   ├── detection.mp4
│   │   └── main.py
│   ├── models
│   │   └── crowdhuman.onnx
│   └── tracking
│       ├── utils.py
│       └── yolov5.py
|
├── docker-compose.yml
├── Dockerfile
├── poetry.lock
├── pyproject.toml
└── README.md
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/I-AM-LAMER/ai_tracking
   cd ai-tracking
   ```

2. Run the application using Docker:
   ```
   docker compose up --build
   ```

3. in the src you see result(number) folder in that folder you will see detection.mp4 - this is tracked video with painted border boxes, prediction_score, tracking_id and people count

## Usage

- Replace (with the same filename) or leave the detection.mp4 file
- Run with docker compose up --build
- Wait for end of a program and observe generated video

## Additional Notes

- The project uses YOLOv5 pretrained on crowdhuman dataset.
- The tracking is focused solely on people detection.
- Optionally, WebSocket can be implemented for enhanced real-time visualization.