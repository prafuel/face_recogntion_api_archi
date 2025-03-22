from fastapi import FastAPI, WebSocket
import cv2
import numpy as np
import base64
import json
import os
import uvicorn

from config import config

from src.utility import parse_model_name
from src.anti_spoof_predict import AntiSpoofPredict

from src.generate_face_image_patches import CropImageForStructureInput

app = FastAPI(title="Liveliness Prediction API")

anti_spoof_predict = AntiSpoofPredict(device_id=0)
image_structure_cropper = CropImageForStructureInput()

async def is_fake_predictions(frame, face_bboxes: list):
    is_fake_results = []
    for bbox in face_bboxes:
        x, y, w, h = bbox
        predictions = np.zeros((1, 3))

        for model_name in os.listdir(config.ANTI_SPOOF_MODELS_DIR):
            h_input, w_input, model_type, scale = parse_model_name(model_name=model_name)
            params = {
                "org_img": frame,
                "bbox": (x, y, w, h),
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }

            if scale is None:
                params['crop'] = False

            crop_img = image_structure_cropper.crop(**params)
            predictions = predictions + anti_spoof_predict.predict(
                crop_img, os.path.join(config.ANTI_SPOOF_MODELS_DIR, model_name)
            )
        
        label = np.argmax(predictions)
        value = predictions[0][label] / 2

        is_fake_results.append({
            "is_fake" : int(label == 2),
            "prob" : round(value, 3),
            "bbox" : bbox
        })

    return is_fake_results

@app.websocket("/ws/is-fake")
async def is_fake_endpoint(websocket: WebSocket):
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

            is_fake_results = await is_fake_predictions(frame, face_bboxes)

            # Return results
            await websocket.send_json({
                "is_fake_results" : is_fake_results
            })
    except Exception as e:
        print(f"Liveness API Error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run("liveliness_api:app", host="0.0.0.0", port=8002)