from pydantic import BaseModel
from typing import List, Tuple

class TrackRequest(BaseModel):
    frame: List[int]  # Assuming the frame is sent as a list of pixel values
    frame_id: int

class TrackResponse(BaseModel):
    tracks: List[Tuple[int, int, int, int]]  # List of bounding boxes (x1, y1, x2, y2)
    frame_id: int