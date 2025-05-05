# People Tracking Project

This project implements a real-time people tracking system using YOLOv8 for detection and DeepSORT for tracking. The application provides an API for tracking video frames and a WebSocket server for real-time visualization of tracking data.

## Project Structure

```
people-tracking-workspace
├── src
│   ├── tracking
│   │   ├── detector.py       # YOLOv8 model for detecting people
│   │   ├── tracker.py        # DeepSORT algorithm for tracking
│   │   └── utils.py          # Utility functions for video processing
│   ├── api
│   │   ├── main.py           # FastAPI application for tracking API
│   │   └── schemas.py        # Pydantic models for API requests/responses
│   ├── visualization
│   │   ├── server.py         # WebSocket server for real-time visualization
│   │   └── static
│   │       ├── index.html     # Main HTML file for client-side application
│   │       └── app.js         # Client-side JavaScript for WebSocket connection
│   └── config.py             # Configuration settings for the application
├── requirements.txt           # Python dependencies
├── docker-compose.yml         # Service definitions for Docker
├── Dockerfile                 # Instructions to build Docker image
├── README.md                  # Project documentation
└── .env                       # Environment variables
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd people-tracking-workspace
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure environment variables in the `.env` file.

4. Run the application using Docker:
   ```
   docker-compose up --build
   ```

5. Access the API at `http://localhost:8000/track` and the visualization at `http://localhost:3000`.

## Usage

- Send video frames to the `/track` endpoint to receive tracking coordinates.
- Connect to the WebSocket server for real-time updates on tracking data.

## Additional Notes

- The project uses YOLOv8 for object detection without additional training.
- The tracking is focused solely on people detection.
- Optionally, WebSocket can be implemented for enhanced real-time visualization.