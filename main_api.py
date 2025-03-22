import asyncio
import base64
import cv2
import json
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from contextlib import AsyncExitStack
import websockets
import uvicorn
from datetime import datetime

from detection.face_detection import (
    get_face_box, blur_image
)

from config import config

app = FastAPI()

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@dataclass
class FaceAnalysisResult:
    bbox: tuple  # (x, y, w, h)
    name: Optional[str] = None
    prob: Optional[float] = None
    is_drowsy: Optional[bool] = None
    concentration_level: Optional[float] = None
    landmarks: Optional[Dict] = None
    emotion: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[int] = None
    is_fake: Optional[bool] = None

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.is_processing = False
        self.selected_features = {
            "blur": config.BLUR,
            "drowsiness": config.DROWSINESS,
            "emotion": config.EMOTION,
            "gender": config.GENDER,
            "age": config.AGE,
            "liveliness": config.LIVELINESS,
            "face_recognition": config.RECOGNITION
        }
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        await websocket.send_json({"type": "features", "data": self.selected_features})
        
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            
    async def update_features(self, features: dict):
        self.selected_features.update(features)
        # Notify all clients about the feature update
        for connection in self.active_connections:
            await connection.send_json({"type": "features", "data": self.selected_features})
            
    async def broadcast_frames(self, original_frame, processed_frame):
        if not self.active_connections:
            return
            
        _, orig_jpeg = cv2.imencode('.jpg', original_frame)
        orig_base64 = base64.b64encode(orig_jpeg).decode('utf-8')
        
        _, proc_jpeg = cv2.imencode('.jpg', processed_frame)
        proc_base64 = base64.b64encode(proc_jpeg).decode('utf-8')
        
        timestamp = datetime.now().timestamp()
        
        message = {
            "type": "frames",
            "original": f"data:image/jpeg;base64,{orig_base64}",
            "processed": f"data:image/jpeg;base64,{proc_base64}",
            "timestamp": timestamp
        }
        
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Error broadcasting to client: {e}")

manager = ConnectionManager()

async def send_frame_to_api(websocket, frame_data):
    await websocket.send(json.dumps(frame_data))
    response = await websocket.recv()
    return json.loads(response)

@app.get("/", response_class=HTMLResponse)
async def get_html():
    with open("static/index.html", "r") as file:
        return file.read()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Start frame processing if it's not already running
        if not manager.is_processing:
            manager.is_processing = True
            asyncio.create_task(process_frames())
            
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                if message.get("type") == "update_features":
                    await manager.update_features(message.get("features", {}))
            except json.JSONDecodeError:
                print("Invalid JSON received")
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)

async def process_frames():
    # Create a video capture object
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    try:
        while len(manager.active_connections) > 0:
            # Use AsyncExitStack to manage multiple websocket connections
            async with AsyncExitStack() as stack:
                # Define the APIs we need to connect to based on selected features
                api_connections = {}
                
                # Connect to APIs based on selected features
                selected_features = manager.selected_features
                
                if selected_features.get("drowsiness", False):
                    api_connections['drowsiness'] = await stack.enter_async_context(
                        websockets.connect(config.DROWSINESS_PREDICTION_WS_API_URL))
                
                if selected_features.get("emotion", False):
                    api_connections['emotion'] = await stack.enter_async_context(
                        websockets.connect(config.EMOTION_PREDICTION_WS_API_URL))
                
                if selected_features.get("gender", False):
                    api_connections['gender'] = await stack.enter_async_context(
                        websockets.connect(config.GENDER_PREDICTION_WS_API_URL))
                
                if selected_features.get("age", False):
                    api_connections['age'] = await stack.enter_async_context(
                        websockets.connect(config.AGE_PREDICTION_WS_API_URL))
                
                if selected_features.get("liveliness", False):
                    api_connections['liveliness'] = await stack.enter_async_context(
                        websockets.connect(config.LIVELINESS_PREDICTION_WS_API_URL))

                if selected_features.get("face_recognition", False):
                    api_connections['recognition'] = await stack.enter_async_context(
                        websockets.connect(config.FACE_RECOGNITION_WS_API_URL))
                
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame - exiting")
                    break

                # Make a copy for display
                display_frame = frame.copy()
                original_frame = frame.copy()
                
                # Get face bounding boxes
                face_bboxes = await get_face_box(frame)
                
                if not face_bboxes:
                    # If no faces detected, just show the frame
                    await manager.broadcast_frames(original_frame, display_frame)
                    await asyncio.sleep(0.033)  # About 30 FPS
                    continue

                # Convert frame to base64
                _, buffer = cv2.imencode('.jpg', frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Create frame data packet
                frame_data = {
                    "frame": frame_base64,
                    "face_bboxes": face_bboxes
                }
                
                # Create a dictionary to store results for each face
                face_results = {tuple(bbox): FaceAnalysisResult(bbox=tuple(bbox)) for bbox in face_bboxes}
                
                # Process all API requests concurrently
                api_tasks = []
                
                # We need liveliness detection for fake face detection
                if selected_features.get("liveliness", False) and 'liveliness' in api_connections:
                    api_tasks.append(send_frame_to_api(api_connections['liveliness'], frame_data))
                
                # Process other APIs only if needed for non-fake faces
                if selected_features.get("drowsiness", False) and 'drowsiness' in api_connections:
                    api_tasks.append(send_frame_to_api(api_connections['drowsiness'], frame_data))

                if selected_features.get("emotion", False) and 'emotion' in api_connections:
                    api_tasks.append(send_frame_to_api(api_connections['emotion'], frame_data))

                if selected_features.get("gender", False) and 'gender' in api_connections:
                    api_tasks.append(send_frame_to_api(api_connections['gender'], frame_data))

                if selected_features.get("age", False) and 'age' in api_connections:
                    api_tasks.append(send_frame_to_api(api_connections['age'], frame_data))

                if selected_features.get("face_recognition", False) and 'recognition' in api_connections:
                    api_tasks.append(send_frame_to_api(api_connections['recognition'], frame_data))
                
                # Wait for all API responses
                if api_tasks:
                    try:
                        api_results = await asyncio.gather(*api_tasks)
                        
                        # Process results
                        result_idx = 0

                        # Process liveliness results first to determine fake faces
                        if selected_features.get("liveliness", False) and 'liveliness' in api_connections:
                            liveness_users = api_results[result_idx]['is_fake_results']
                            for user in liveness_users:
                                bbox = tuple(user['bbox'])
                                if bbox in face_results:
                                    face_results[bbox].is_fake = user.get('is_fake')
                            result_idx += 1
                        
                        # Process other results only if needed
                        if selected_features.get("drowsiness", False) and 'drowsiness' in api_connections:
                            drowsiness_users = api_results[result_idx]['drowsiness_results']
                            for user in drowsiness_users:
                                bbox = tuple(user['bbox'])
                                if bbox in face_results:
                                    face_results[bbox].is_drowsy = user.get('is_drowsy')
                                    face_results[bbox].concentration_level = user.get('concentration_level')
                                    face_results[bbox].landmarks = user.get('landmarks')
                            result_idx += 1


                        if selected_features.get("emotion", False) and 'emotion' in api_connections:
                            emotion_users = api_results[result_idx]['users_emotion']
                            for user in emotion_users:
                                bbox = tuple(user['bbox'])
                                if bbox in face_results:
                                    face_results[bbox].emotion = user.get('emotion')
                            result_idx += 1

                        if selected_features.get("gender", False) and 'gender' in api_connections:
                            gender_users = api_results[result_idx]['gender_results']
                            for user in gender_users:
                                bbox = tuple(user['bbox'])
                                if bbox in face_results:
                                    face_results[bbox].gender = user.get('gender')
                            result_idx += 1

                        if selected_features.get("age", False) and 'age' in api_connections:
                            age_users = api_results[result_idx]['age_results']
                            for user in age_users:
                                bbox = tuple(user['bbox'])
                                if bbox in face_results:
                                    face_results[bbox].age = user.get('age')
                            result_idx += 1

                        if selected_features.get("face_recognition", False) and 'recognition' in api_connections:
                            recognized_users = api_results[result_idx]['recognized_users']
                            for user in recognized_users:
                                bbox = tuple(user['bbox'])
                                if bbox in face_results:
                                    face_results[bbox].name = user.get('name', 'unknown')
                                    face_results[bbox].prob = user.get('prob')
                                    face_results[bbox].age = user.get("age") if user.get("age") else face_results[bbox].age
                                    face_results[bbox].gender = user.get("gender") if user.get("gender") else face_results[bbox].gender
                            result_idx += 1
                    except Exception as e:
                        print(f"Error processing API results: {e}")

                # Draw results on frame
                display_frame = await draw_results_on_frame(display_frame, face_results, selected_features)
                
                # Send frames to all connected clients
                await manager.broadcast_frames(original_frame, display_frame)
                
                # Sleep to control frame rate
                await asyncio.sleep(0.033)  # About 30 FPS
                
    except Exception as e:
        print(f"Error in frame processing: {e}")
    finally:
        # Release resources
        cap.release()
        manager.is_processing = False

async def draw_results_on_frame(frame, face_results, selected_features):
    """
    Draw analysis results on the frame.
    For fake faces, only draw a red box with no additional information.
    For real faces, draw a green box with information.
    """
    result_frame = frame.copy()
    
    for face_data in face_results.values():
        x, y, w, h = face_data.bbox

        # Blur Face if enabled
        if selected_features.get("blur", False):
            result_frame = await blur_image(result_frame, [x, y, w, h], blur_strength=200)
        
        # Check if face is detected as fake
        if face_data.is_fake and selected_features.get("liveliness", False):
            # For fake faces, only draw a red rectangle
            cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 0, 255), 3)  # Thicker red box
            
            # Add "FAKE" text above the box
            # cv2.putText(result_frame, "FAKE", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        else:
            # For real faces, draw green rectangle
            cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Calculate initial text position
            text_y = y - 10
        
            # Add text for each available piece of information based on selected features
            if face_data.name and selected_features.get("face_recognition", False):
                text_y -= 20
                cv2.putText(result_frame, f"NAME: {face_data.name} ({face_data.prob:.2f})", 
                            (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if face_data.age is not None and selected_features.get("age", False):
                text_y -= 20
                cv2.putText(result_frame, f"AGE: {face_data.age}", 
                            (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if face_data.gender and selected_features.get("gender", False):
                text_y -= 20
                cv2.putText(result_frame, f"GENDER: {face_data.gender}", 
                            (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if face_data.emotion and selected_features.get("emotion", False):
                text_y -= 20
                cv2.putText(result_frame, f"EMOTION: {face_data.emotion}", 
                            (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if face_data.is_drowsy is not None and selected_features.get("drowsiness", False):
                text_y -= 20
                cv2.putText(result_frame, f"DROWSY: {face_data.is_drowsy}", 
                            (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                if face_data.concentration_level is not None:
                    text_y -= 20
                    cv2.putText(result_frame, f"CONCENTRATION: {face_data.concentration_level}", 
                                (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Draw facial landmarks if available and drowsiness detection is enabled
            if face_data.landmarks and selected_features.get("drowsiness", False):
                landmarks = face_data.landmarks
                if "left_eye" in landmarks and "right_eye" in landmarks:
                    leftEyeHull = cv2.convexHull(np.array(landmarks["left_eye"]))
                    rightEyeHull = cv2.convexHull(np.array(landmarks["right_eye"]))
                    cv2.drawContours(result_frame, [leftEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(result_frame, [rightEyeHull], -1, (0, 255, 0), 1)
                
                if "mouth" in landmarks:
                    mouthHull = cv2.convexHull(np.array(landmarks["mouth"]))
                    cv2.drawContours(result_frame, [mouthHull], -1, (0, 0, 255), 1)
    
    return result_frame

if __name__ == "__main__":
    uvicorn.run("main_api:app", host="0.0.0.0", port=8000, reload=True)