from fastapi import FastAPI, WebSocket
import cv2
import numpy as np
import base64
import json
import os
import uvicorn

from config import config

app = FastAPI(title="Age Prediction API")

age_net = cv2.dnn.readNet(config.AGE_MODEL, config.AGE_PROTO)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

async def age_predictions(frame, face_bboxes: list):
    age_results = []
    for bbox in face_bboxes:
        x, y, w, h = bbox
        # h, w, _ = frame.shape
        # x, y, w, h = max(0, x), max(0, y), min(w, w - x), min(h, h - y)

        face_img = frame[y:y+h, x:x+w].copy()

        if face_img.size == 0:
            print("Warning: Empty cropped face image!")
            age = "Analyzing..."
        else:
            # Convert face image to blob
            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), 
                                        (78.4263377603, 87.7689143744, 114.895847746), 
                                        swapRB=False)

            # Predict Age
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]

        age_results.append({
            "age" : age,
            "bbox" : [x, y, w, h]
        })
    return age_results

@app.websocket("/ws/get-age")
async def age_prediction_endpoint(websocket: WebSocket):
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

            age_results = await age_predictions(frame, face_bboxes)

            # Return results
            await websocket.send_json({
                "age_results" : age_results
            })
    except Exception as e:
        print(f"AGE Prediction API Error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run("age_prediction_api:app", host="0.0.0.0", port=8003)