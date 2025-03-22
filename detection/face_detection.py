
import mediapipe as mp
import cv2

from typing import List

# Face Detection model
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

async def get_face_box(frame) -> List:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    all_faces_in_frame = face_detection.process(rgb_frame)

    bboxes = []
    if all_faces_in_frame.detections:
        for face in all_faces_in_frame.detections:
            bboxC = face.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, width, height = (int(bboxC.xmin * w), int(bboxC.ymin * h),
                                    int(bboxC.width * w), int(bboxC.height * h))
            
            # face_vector = get_face_vectors(self.model, frame, [x, y, width, height])
            points = [x, y, width, height]
            # Face Square
            # cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

            bboxes.append(points)

    return bboxes

async def blur_image(image, points, blur_strength=30):
    """
    Blur a specific region of an image defined by points [x, y, w, h].
    
    Args:
        image: Input image (numpy array)
        points: List or tuple containing [x, y, w, h] coordinates of region to blur
        blur_strength: Blur kernel size (default: 30)
        
    Returns:
        Modified image with blurred region
    """
    # Handle invalid input
    if image is None or len(image.shape) < 2:
        raise ValueError("Invalid image input")
        
    if not isinstance(points, (list, tuple)) or len(points) != 4:
        raise ValueError("Points must be a list or tuple [x, y, w, h]")
    
    # Create a copy to avoid modifying the original
    result = image.copy()
    h_img, w_img = result.shape[:2]  # Get image dimensions
    x, y, w, h = points
    
    # Ensure coordinates are integers
    x, y, w, h = int(x), int(y), int(w), int(h)
    
    # Ensure x, y, w, h are within bounds
    x = max(0, min(x, w_img - 1))
    y = max(0, min(y, h_img - 1))
    w = max(1, min(w, w_img - x))  # Ensure width is at least 1
    h = max(1, min(h, h_img - y))  # Ensure height is at least 1
    
    # Adjust blur kernel size based on ROI size for better performance
    blur_size = min(blur_strength, max(3, min(w // 2, h // 2) * 2 - 1))
    # Ensure kernel size is odd
    if blur_size % 2 == 0:
        blur_size -= 1
    
    try:
        # Extract region of interest
        roi = result[y:y+h, x:x+w]
        
        # Apply Gaussian Blur instead of simple blur for better quality
        blurred_roi = cv2.GaussianBlur(roi, (blur_size, blur_size), 0)
        
        # Replace the original region with blurred region
        result[y:y+h, x:x+w] = blurred_roi
        
    except Exception as e:
        print(f"Error during image blurring: {e}")
        # Return original image instead of fully blurred image on error
        return image
    
    return result