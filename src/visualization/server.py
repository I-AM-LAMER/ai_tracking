from fastapi import WebSocket, APIRouter
from fastapi.responses import HTMLResponse
import json
import os

router = APIRouter()

@router.get("/")
async def get():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, '..', 'src', 'visualization', 'static', 'index.html')
    MODEL_PATH = os.path.abspath(MODEL_PATH)
    return HTMLResponse(open(MODEL_PATH).read())
