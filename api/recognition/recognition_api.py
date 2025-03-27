# from fastapi import FastAPI, WebSocket
# import cv2
# import numpy as np
# import base64
# import json
# import uvicorn
# from config import config

# from keras_vggface.vggface import VGGFace
# from src.utility import identify_user, get_face_vector

# app = FastAPI(title="Face Recognition API")

# # Loading Model for face_vectors
# model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
# print("Model load...")

# # Loading Database
# # npy_database = np.load(config.DETAILED_NPY_FILE, allow_pickle=True).item()
# # face_vectors = npy_database['face_vectors']
# # labels = npy_database['labels']
# # user_age_data = npy_database['age']
# # user_gender_data = npy_database['gender']

# print("Database load...")

# threshold = config.THRESHOLD

# async def recognize_faces(frame, face_bboxes: list, user_cache: dict) -> list:
#     recognized_users = []
#     for bbox in face_bboxes:
#         x, y, w, h = bbox
        
#         name = "unknown"
#         print("*"*100)
        
#         # Get face vector
#         try:
#             face_vector = get_face_vector(model=model, frame=frame, points=bbox)
            
#             # Handle case where face_vector might be None
#             if face_vector is None:
#                 continue
                
#             print("face_vector shape:", face_vector.shape)
            
#             # Check if we have faces in the cache
#             if len(user_cache['face_vectors']) > 0:
#                 # Convert cache vectors to numpy array if needed
#                 cache_vectors = np.array(user_cache['face_vectors'])
                
#                 # Reshape if needed
#                 if len(cache_vectors.shape) == 3:
#                     cache_vectors = np.vstack(cache_vectors)
                
#                 max_prob, index = identify_user(
#                     face_vector=face_vector,
#                     trained_face_vectors=cache_vectors
#                 )
#                 # For Cache
#                 name = user_cache['labels'][index] if max_prob > threshold else "unknown"
#                 age = user_cache['age'][index] if max_prob > threshold else False
#                 gender = user_cache['gender'][index] if max_prob > threshold else False

#                 if name == "unknown":
#                     # No cache, use main database
#                     max_prob, index = identify_user(
#                         face_vector=face_vector,
#                         trained_face_vectors=face_vectors
#                     )
                    
#                     name = labels[index] if max_prob > threshold else "unknown"
#                     age = user_age_data[index] if max_prob > threshold else False
#                     gender = user_gender_data[index] if max_prob > threshold else False
            
#             else:
#                 # No cache, use main database
#                 max_prob, index = identify_user(
#                     face_vector=face_vector,
#                     trained_face_vectors=face_vectors
#                 )
                
#                 name = labels[index] if max_prob > threshold else "unknown"
#                 age = user_age_data[index] if max_prob > threshold else False
#                 gender = user_gender_data[index] if max_prob > threshold else False
        
#             recognized_users.append({
#                 "bbox": bbox,
#                 "prob": round(float(max_prob), 3),
#                 "name": name,
#                 "age": age,
#                 "gender": gender
#             })
            
#             # Add to cache if not already there and not unknown
#             if name not in user_cache['labels'] and name != "unknown":
#                 user_cache['face_vectors'].append(face_vector)
#                 user_cache['labels'].append(name)
#                 user_cache['age'].append(age)
#                 user_cache['gender'].append(gender)

#             print("user cache : ", user_cache)
        
#         except Exception as e:
#             print(f"Error processing face: {e}")
#             continue
    
#     return recognized_users, user_cache


# @app.websocket("/ws/recognize")
# async def recognition_endpoint(websocket: WebSocket):
#     user_cache = {
#         "face_vectors": [],
#         "labels": [],
#         "age": [],
#         "gender": []
#     }
    
#     await websocket.accept()
#     try:
#         while True:
#             # Receive JSON with frame and bbox data
#             data = await websocket.receive_json()
            
#             # Decode base64 image
#             frame_data = data.get("frame", "")
#             jpg_original = base64.b64decode(frame_data)
#             jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
#             frame = cv2.imdecode(jpg_as_np, flags=1)
            
#             # Get face bounding boxes
#             face_bboxes = data.get("face_bboxes", [])
            
#             # Check for Cache
#             recognized_users, user_cache = await recognize_faces(
#                 frame=frame,
#                 face_bboxes=face_bboxes,
#                 user_cache=user_cache
#             )
            
#             print("User Data : ", user_cache['labels'])
            
#             # Return results
#             await websocket.send_json({
#                 "recognized_users": recognized_users
#             })
#     except Exception as e:
#         print(f"Recognition API Error: {e}")
#     finally:
#         await websocket.close()

# if __name__ == "__main__":
#     uvicorn.run("recognition_api:app", host="0.0.0.0", port=8001)


# new code
from fastapi import FastAPI, WebSocket
import cv2
import numpy as np
import base64
import json
import uvicorn
from config import config

from keras_vggface.vggface import VGGFace
from src.utility import identify_user, get_face_vector

app = FastAPI(title="Face Recognition API")

# Loading Model for face_vectors
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
print("Model load...")

# Threshold for face recognition
threshold = config.THRESHOLD

async def recognize_faces(frame, face_bboxes: list, user_cache: dict) -> list:
    recognized_users = []
    for bbox in face_bboxes:
        x, y, w, h = bbox
        
        name = "unknown"
        print("*"*100)
        
        # Get face vector
        try:
            face_vector = get_face_vector(model=model, frame=frame, points=bbox)
            
            # Handle case where face_vector might be None
            if face_vector is None:
                continue
                
            print("face_vector shape:", face_vector.shape)
            
            # Use the new Milvus-based identification
            distance, name = identify_user(face_vector)
            
            # If no match found in Milvus or distance is too high
            if name is None or distance > threshold:
                name = "unknown"
            
            recognized_users.append({
                "bbox": bbox,
                "prob": round(float(distance), 3),
                "name": name
            })
            
            # Optionally add to cache if not already there and not unknown
            if name not in user_cache['labels'] and name != "unknown":
                user_cache['labels'].append(name)

            print("user cache : ", user_cache)
        
        except Exception as e:
            print(f"Error processing face: {e}")
            continue
    
    return recognized_users, user_cache


@app.websocket("/ws/recognize")
async def recognition_endpoint(websocket: WebSocket):
    user_cache = {
        "labels": []
    }
    
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
            
            # Recognize faces
            recognized_users, user_cache = await recognize_faces(
                frame=frame,
                face_bboxes=face_bboxes,
                user_cache=user_cache
            )
            
            print("User Data : ", user_cache['labels'])
            
            # Return results
            await websocket.send_json({
                "recognized_users": recognized_users
            })
    except Exception as e:
        print(f"Recognition API Error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run("recognition_api:app", host="0.0.0.0", port=8001)