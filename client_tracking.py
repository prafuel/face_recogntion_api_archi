from detection.face_detection import get_face_box
from tracking.tracking import create_tracker, track_objects
import asyncio
import cv2
import time
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

tracker = create_tracker()

def draw_results(frame, tracked_objects):
    """Draw bounding boxes for tracked objects on the frame"""
    if not tracked_objects:
        return
        
    for tracked_obj in tracked_objects:
        # Get the ID for display
        obj_id = tracked_obj.id
        
        # Extract points from the tracked object's estimate
        points = tracked_obj.estimate
        
        # Check if there's detection data with bounding box
        if hasattr(tracked_obj, 'last_detection') and tracked_obj.last_detection.data:
            # Get the original bounding box
            x, y, w, h = tracked_obj.last_detection.data["bbox"]
            
            # Draw rectangle with ID
            cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {obj_id}", (int(x), int(y-10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # If we only have centroid, draw a circle
            point_x, point_y = points[0]
            cv2.circle(frame, (int(point_x), int(point_y)), 10, (0, 0, 255), -1)
            cv2.putText(frame, f"ID: {obj_id}", (int(point_x), int(point_y-10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


async def face_object_tracking():
    total_frames = 0
    skip_frames = 2  # Process every 3rd frame with face detection
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            total_frames += 1
            
            # Run face detection only on specific frames
            faces = await get_face_box(frame)
            if total_frames % (skip_frames + 1) == 0:
                time.sleep(0.4)
                # Draw the detected face boxes directly for debugging
                for face in faces:
                    x, y, w, h = face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(frame, "Detected", (x, y-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Update tracker with new detections
                if faces:
                    tracked_objects = track_objects(faces, tracker)
                    draw_results(frame, tracked_objects)
            else:
                # Just update existing tracks on skipped frames
                tracked_objects = tracker.update([])
                draw_results(frame, tracked_objects)
            
            # Display frame count
            cv2.putText(frame, f"Frame: {total_frames}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("Face Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print(f"Total frames processed: {total_frames}")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(face_object_tracking())