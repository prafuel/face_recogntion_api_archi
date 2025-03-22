from fastapi import FastAPI, WebSocket
import uvicorn

import numpy as np
import base64
import cv2

from config import config

from src.utility import (
    get_emotion
)


app = FastAPI(title="Emotion Prediction API")

async def predict_emotion(frame, face_bboxes: list):
    users_emotions = []
    for bbox in face_bboxes:
        current_emotion = get_emotion(frame, bbox)
        users_emotions.append({
            "emotion" : current_emotion,
            "bbox" : bbox
        })

    return users_emotions

@app.websocket("/ws/get-emotion")
async def emotion_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()
            frame_data = data.get("frame", "")

            jpg_original = base64.b64decode(frame_data)
            jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
            frame = cv2.imdecode(jpg_as_np, flags=1)
            
            # Get face bounding boxes
            face_bboxes = data.get("face_bboxes", [])

            await websocket.send_json({
                "users_emotion" : await predict_emotion(frame, face_bboxes)
            })


    except Exception as e:
        print(f"Error in Emotion Detection : {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run("emotion_api:app", host="0.0.0.0", port=8005)