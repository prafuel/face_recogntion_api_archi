import asyncio
import base64
import cv2
import json
import numpy as np
import websockets
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from contextlib import AsyncExitStack

from detection.face_detection import (
    get_face_box, blur_image
)

from config import config

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


async def send_frame_to_api(websocket, frame_data):
    await websocket.send(json.dumps(frame_data))
    response = await websocket.recv()
    print("json_output", json.loads(response))
    return json.loads(response)


async def process_frames():
    # Use AsyncExitStack to manage multiple websocket connections
    async with AsyncExitStack() as stack:
        # Define the APIs we need to connect to based on config
        api_connections = {}
        if config.DROWSINESS:
            api_connections['drowsiness'] = await stack.enter_async_context(
                websockets.connect(config.DROWSINESS_PREDICTION_WS_API_URL))
            
            
        if config.EMOTION:
            api_connections['emotion'] = await stack.enter_async_context(
                websockets.connect(config.EMOTION_PREDICTION_WS_API_URL))
            
        if config.GENDER:
            api_connections['gender'] = await stack.enter_async_context(
                websockets.connect(config.GENDER_PREDICTION_WS_API_URL))
            
        if config.AGE:
            api_connections['age'] = await stack.enter_async_context(
                websockets.connect(config.AGE_PREDICTION_WS_API_URL))
            
        if config.LIVELINESS:
            api_connections['liveliness'] = await stack.enter_async_context(
                websockets.connect(config.LIVELINESS_PREDICTION_WS_API_URL))

        if config.RECOGNITION:
            api_connections['recognition'] = await stack.enter_async_context(
                websockets.connect(config.FACE_RECOGNITION_WS_API_URL))
            
        # Set up video capture with proper resolution
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        
        try:
            while True:
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame - exiting")
                    break

                # Display the frame even if we're skipping analysis
                display_frame = frame.copy()
                
                # Get face bounding boxes
                face_bboxes = await get_face_box(frame)
                
                if not face_bboxes:
                    # If no faces detected, just show the frame
                    cv2.imshow('Frame', display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
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
                if config.LIVELINESS and 'liveliness' in api_connections:
                    api_tasks.append(send_frame_to_api(api_connections['liveliness'], frame_data))
                
                # Process other APIs only if needed for non-fake faces
                if config.DROWSINESS and 'drowsiness' in api_connections:
                    api_tasks.append(send_frame_to_api(api_connections['drowsiness'], frame_data))

                if config.EMOTION and 'emotion' in api_connections:
                    api_tasks.append(send_frame_to_api(api_connections['emotion'], frame_data))

                if config.GENDER and 'gender' in api_connections:
                    api_tasks.append(send_frame_to_api(api_connections['gender'], frame_data))

                if config.AGE and 'age' in api_connections:
                    api_tasks.append(send_frame_to_api(api_connections['age'], frame_data))

                if config.RECOGNITION and 'recognition' in api_connections:
                    api_tasks.append(send_frame_to_api(api_connections['recognition'], frame_data))
                
                # Wait for all API responses
                if api_tasks:
                    api_results = await asyncio.gather(*api_tasks)
                    
                    # Process results
                    result_idx = 0

                    # Process liveliness results first to determine fake faces
                    if config.LIVELINESS and 'liveliness' in api_connections:
                        liveness_users = api_results[result_idx]['is_fake_results']
                        for user in liveness_users:
                            bbox = tuple(user['bbox'])
                            if bbox in face_results:
                                face_results[bbox].is_fake = user.get('is_fake')
                        result_idx += 1
                    
                    # Process other results only if needed
                    if config.DROWSINESS and 'drowsiness' in api_connections:
                        drowsiness_users = api_results[result_idx]['drowsiness_results']
                        for user in drowsiness_users:
                            bbox = tuple(user['bbox'])
                            if bbox in face_results:
                                face_results[bbox].is_drowsy = user.get('is_drowsy')
                                face_results[bbox].concentration_level = user.get('concentration_level')
                                face_results[bbox].landmarks = user.get('landmarks')
                        result_idx += 1


                    if config.EMOTION and 'emotion' in api_connections:
                        emotion_users = api_results[result_idx]['users_emotion']
                        for user in emotion_users:
                            bbox = tuple(user['bbox'])
                            if bbox in face_results:
                                face_results[bbox].emotion = user.get('emotion')
                        result_idx += 1

                    if config.GENDER and 'gender' in api_connections:
                        gender_users = api_results[result_idx]['gender_results']
                        for user in gender_users:
                            bbox = tuple(user['bbox'])
                            if bbox in face_results:
                                face_results[bbox].gender = user.get('gender')
                        result_idx += 1

                    if config.AGE and 'age' in api_connections:
                        age_users = api_results[result_idx]['age_results']
                        for user in age_users:
                            bbox = tuple(user['bbox'])
                            if bbox in face_results:
                                face_results[bbox].age = user.get('age')
                        result_idx += 1

                    if config.RECOGNITION and 'recognition' in api_connections:
                        recognized_users = api_results[result_idx]['recognized_users']
                        for user in recognized_users:
                            bbox = tuple(user['bbox'])

                            print("User : ", user)
                            if bbox in face_results:
                                face_results[bbox].name = user.get('name', 'unknown')
                                face_results[bbox].prob = user.get('prob')
                                face_results[bbox].age = user.get("age") if user.get("age") else face_results[bbox].age
                                face_results[bbox].gender = user.get("gender") if user.get("gender") else face_results[bbox].gender
                                
                        result_idx += 1

                # Draw results on frame
                display_frame = await draw_results_on_frame(display_frame, face_results)
                
                # Display the frame
                cv2.imshow('Frame', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            # Ensure resources are properly released
            cap.release()
            cv2.destroyAllWindows()


async def draw_results_on_frame(frame, face_results):
    """
    Draw analysis results on the frame.
    For fake faces, only draw a red box with no additional information.
    For real faces, draw a green box with information.
    """
    result_frame = frame.copy()
    
    for face_data in face_results.values():
        x, y, w, h = face_data.bbox

        # Blur Face
        if config.BLUR:
                result_frame = await blur_image(result_frame, [x, y, w, h], blur_strength=200)
        
        # Check if face is detected as fake
        if face_data.is_fake:
            # For fake faces, only draw a red rectangle
            cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 0, 255), 3)  # Thicker red box
            
            # Add "FAKE" text above the box
            # cv2.putText(result_frame, "FAKE", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        else:
            # For real faces, draw green rectangle
            cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Calculate initial text position
            text_y = y - 10
        
            # Add text for each available piece of information
            if face_data.name:
                text_y -= 20
                cv2.putText(result_frame, f"NAME: {face_data.name} ({face_data.prob:.2f})", 
                            (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if face_data.age is not None and config.AGE:
                text_y -= 20
                cv2.putText(result_frame, f"AGE: {face_data.age}", 
                            (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if face_data.gender and config.GENDER:
                text_y -= 20
                cv2.putText(result_frame, f"GENDER: {face_data.gender}", 
                            (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if face_data.emotion:
                text_y -= 20
                cv2.putText(result_frame, f"EMOTION: {face_data.emotion}", 
                            (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if face_data.is_drowsy is not None:
                text_y -= 20
                cv2.putText(result_frame, f"DROWSY: {face_data.is_drowsy}", 
                            (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                if face_data.concentration_level is not None:
                    text_y -= 20
                    cv2.putText(result_frame, f"CONCENTRATION: {face_data.concentration_level}", 
                                (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Draw facial landmarks if available
            if face_data.landmarks:
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
    asyncio.run(process_frames())