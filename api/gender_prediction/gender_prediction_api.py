from fastapi import FastAPI, WebSocket
import cv2
import numpy as np
import base64
import json
import os
import uvicorn

from config import config

app = FastAPI(title="Gender Prediction API")

gender_net = cv2.dnn.readNet(config.GENDER_MODEL, config.GENDER_PROTO)
gender_list = ['Male', 'Female']

async def gender_predictions(frame, face_bboxes: list):
    gender_results = []
    for bbox in face_bboxes:
        x, y, w, h = bbox
        # h, w, _ = frame.shape
        # x, y, w, h = max(0, x), max(0, y), min(w, w - x), min(h, h - y)

        face_img = frame[y:y+h, x:x+w].copy()

        if face_img.size == 0:
            print("Warning: Empty cropped face image!")
            gender = "Analyzing..."
        else:
            # Convert face image to blob
            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), 
                                        (78.4263377603, 87.7689143744, 114.895847746), 
                                        swapRB=False)

            # Predict Age
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]

        gender_results.append({
            "gender" : gender,
            "bbox" : [x, y, w, h]
        })
    return gender_results

@app.websocket("/ws/get-gender")
async def gender_prediction_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive JSON with frame and bbox data
            data = await websocket.receive_json()
            
            # Decode base64 image
            frame_data = data.get("frame", "")
            jpg_original = base64.b64decode(frame_data)
            jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
            frame = cv2.imdecode(jpg_as_np, flags=1)
            
            # Get face bounding boxes
            face_bboxes = data.get("face_bboxes", [])

            gender_results = await gender_predictions(frame, face_bboxes)

            # Return results
            await websocket.send_json({
                "gender_results" : gender_results
            })
    except Exception as e:
        print(f"Gender Prediction API Error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run("gender_prediction_api:app", host="0.0.0.0", port=8004)