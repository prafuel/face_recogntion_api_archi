import cv2
import dlib
from imutils import face_utils
import time
import numpy as np
import base64
import json
from fastapi import FastAPI, WebSocket
import uvicorn

from config import config

from src.utility import (
    eye_aspect_ratio,
    mouth_aspect_ratio,
    head_movement
)

app = FastAPI(title="Emotion Prediction API")

# Load the face detector and landmark predictor once at startup
dlib_face_detector = dlib.get_frontal_face_detector()
dlib_face_landmark_predictor = dlib.shape_predictor(config.DLIB_FACE_LANDMARK_PREDICTOR_MODEL_PATH)

# Thresholds
EYE_AR_THRESH = 0.25      # Eye aspect ratio threshold for closed eyes
EYE_AR_CONSEC_FRAMES = 5  # Number of consecutive frames for drowsiness detection
MOUTH_AR_THRESH = 0.6     # Mouth aspect ratio threshold for yawning
MOUTH_AR_CONSEC_FRAMES = 5  # Consecutive frames for yawn detection
HEAD_MOVEMENT_THRESH = 5  # Threshold for head movement detection
BLINK_PATTERN_THRESH = 5  # Number of blinks to check for pattern

# Create global state dictionaries to track state across frames for each face
drowsiness_state = {}

async def get_drowsiness(frame, face_bboxes: list):
    global drowsiness_state
    drowsiness_results = []
    
    # Process each detected face
    for i, bbox in enumerate(face_bboxes):
        face_id = f"face_{i}"  # Simple face ID based on position
        
        # Initialize or get state for this face
        if face_id not in drowsiness_state:
            drowsiness_state[face_id] = {
                "counter": 0,
                "yawn_counter": 0,
                "blink_counter": 0,
                "distraction_counter": 0,
                "blink_pattern": [],
                "prev_landmarks": None,
                "total_blinks": 0,
                "total_yawns": 0,
                "last_blink_time": time.time()
            }
        
        state = drowsiness_state[face_id]
        
        # Extract face coordinates
        x, y, w, h = bbox
        
        # Convert to grayscale for dlib
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Convert bbox to dlib rectangle
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        
        try:
            # Get facial landmarks
            shape = dlib_face_landmark_predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            
            # Extract eye and mouth coordinates
            (l_start, l_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
            (r_start, r_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
            (mouth_start, mouth_end) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
            
            leftEye = shape[l_start:l_end]
            rightEye = shape[r_start:r_end]
            mouth = shape[mouth_start:mouth_end]
            
            # Calculate Eye Aspect Ratio (EAR)
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            earAvg = (leftEAR + rightEAR) / 2.0
            
            # Calculate Mouth Aspect Ratio (MAR) for yawn detection
            mar = mouth_aspect_ratio(mouth)
            
            # Initialize result dictionary with default values
            result = {
                "is_drowsy": "No",  # Default to No
                "concentration_level": "Normal",
                "extra_info": [],
                "metrics": {
                    "ear": float(earAvg),
                    "mar": float(mar),
                    "blinks": state["total_blinks"],
                    "yawns": state["total_yawns"]
                },
                "landmarks": {
                    "left_eye": leftEye.tolist(),
                    "right_eye": rightEye.tolist(),
                    "mouth": mouth.tolist()
                },
                "bbox": bbox
            }
            
            # Detect drowsiness by checking if eyes are closed
            if earAvg < EYE_AR_THRESH:
                state["counter"] += 1
                
                # If eyes are closed for sufficient frames, alert for drowsiness
                if state["counter"] >= EYE_AR_CONSEC_FRAMES:
                    result["is_drowsy"] = "Yes"
                    result["extra_info"].append("Drowsiness detected")
            else:
                # Reset counter if eyes are open but count it as a blink if counter reached threshold
                if 3 <= state["counter"] < EYE_AR_CONSEC_FRAMES:  # Minimum frames to consider it a blink
                    state["total_blinks"] += 1
                    # Record time between blinks
                    current_time = time.time()
                    if state["last_blink_time"] > 0:
                        state["blink_pattern"].append(current_time - state["last_blink_time"])
                    state["last_blink_time"] = current_time
                    # Update blink count in the result
                    result["metrics"]["blinks"] = state["total_blinks"]
                
                state["counter"] = 0
            
            # Detect yawning
            if mar > MOUTH_AR_THRESH:
                state["yawn_counter"] += 1
                if state["yawn_counter"] >= MOUTH_AR_CONSEC_FRAMES:
                    # Only increment total_yawns if a new yawn is detected
                    if state["yawn_counter"] == MOUTH_AR_CONSEC_FRAMES:
                        state["total_yawns"] += 1
                        result["metrics"]["yawns"] = state["total_yawns"]
                    
                    result["extra_info"].append("Yawning detected")
                    if result["is_drowsy"] == "No":
                        result["is_drowsy"] = "Possible"
            else:
                state["yawn_counter"] = 0
            
            # Detect head movement/distraction
            if state["prev_landmarks"] is not None:
                movement = head_movement(state["prev_landmarks"], shape)
                
                if movement > HEAD_MOVEMENT_THRESH:
                    state["distraction_counter"] += 1
                    if state["distraction_counter"] > 10:
                        result["extra_info"].append("Distraction detected")
                        result["concentration_level"] = "Distracted"
                else:
                    state["distraction_counter"] = max(0, state["distraction_counter"] - 1)
            
            # Update landmarks for next frame comparison
            state["prev_landmarks"] = shape.copy()
            
            # Analyze blink patterns for concentration level
            time_since_last_blink = time.time() - state["last_blink_time"]
            
            # Limit the pattern list size
            if len(state["blink_pattern"]) > 20:
                state["blink_pattern"].pop(0)
            
            # Determine concentration level based on blink pattern
            if len(state["blink_pattern"]) >= BLINK_PATTERN_THRESH:
                if np.mean(state["blink_pattern"]) < 0.4:  # Rapid blinking
                    result["concentration_level"] = "Low (Rapid Blinking)"
                elif time_since_last_blink > 7.0:  # Infrequent blinking
                    result["concentration_level"] = "Intense Focus"
            
            drowsiness_results.append(result)
        
        except Exception as e:
            print(f"Error processing face {face_id}: {e}")
            # Return a basic error result
            drowsiness_results.append({
                "is_drowsy": "Unknown",
                "concentration_level": "Unknown",
                "extra_info": [f"Error: {str(e)}"],
                "bbox": bbox
            })
    
    # Clean up stale faces that weren't detected in this frame
    if len(face_bboxes) > 0:
        face_ids = [f"face_{i}" for i in range(len(face_bboxes))]
        for face_id in list(drowsiness_state.keys()):
            if face_id not in face_ids:
                del drowsiness_state[face_id]
    
    return drowsiness_results

@app.websocket("/ws/drowsiness")
async def drowsiness_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive JSON with frame and bbox data
            data = await websocket.receive_json()
            
            try:
                # Decode base64 image
                frame_data = data.get("frame", "")
                if not frame_data:
                    await websocket.send_json({"error": "No frame data provided"})
                    continue
                
                jpg_original = base64.b64decode(frame_data)
                jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
                frame = cv2.imdecode(jpg_as_np, flags=1)
                
                if frame is None:
                    await websocket.send_json({"error": "Failed to decode image"})
                    continue
                
                # Get face bounding boxes
                face_bboxes = data.get("face_bboxes", [])
                if not face_bboxes:
                    await websocket.send_json({"drowsiness_results": [], "message": "No faces detected"})
                    continue

                drowsiness_results = await get_drowsiness(frame, face_bboxes)
                
                # Convert any numpy types to native Python types for JSON serialization
                for result in drowsiness_results:
                    for key, value in result["metrics"].items():
                        if isinstance(value, np.number):
                            result["metrics"][key] = float(value)

                # Return results
                await websocket.send_json({
                    "drowsiness_results": drowsiness_results
                })

            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON data"})
            except Exception as e:
                print(f"Processing error: {e}")
                await websocket.send_json({"error": f"Processing error: {str(e)}"})

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run("drowsiness_api:app", host="0.0.0.0", port=8006)