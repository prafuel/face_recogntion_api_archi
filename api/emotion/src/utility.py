import cv2
import numpy as np

from deepface import DeepFace

def crop_image(image, points: list, image_shape=(224, 224)):
    x, y, w, h = points

    # Check if image is loaded
    if image is None:
        return None
        # raise ValueError("Image is None, check if it's loaded correctly.")
    
    print("image: ", image.shape)

    # Check if points are within the image bounds
    h_img, w_img, _ = image.shape
    if x < 0 or y < 0 or x + w > w_img or y + h > h_img:
        return None
        # raise ValueError(f"Invalid cropping coordinates: {points}, Image shape: {image.shape}")

    crop_face = image[y:y + h, x:x + w]

    # Ensure crop_face is not empty
    if crop_face.size == 0:
        return None
        # raise ValueError("Cropped face is empty, check the coordinates.")

    # Resize to 224x224
    crop_face = cv2.resize(crop_face, image_shape)

    # Expand dimensions for model input
    # crop_face = np.expand_dims(crop_face, axis=0)

    return crop_face

def get_emotion(frame, points):
        x, y, width, height = points
        cropped_image = crop_image(frame, [x, y, width, height])
        if isinstance(cropped_image, np.ndarray):
            try:
                deepface_analysis = DeepFace.analyze(
                    cropped_image,
                    actions=['emotion'],
                    enforce_detection=False
                )
                
                if isinstance(deepface_analysis, list):
                    deepface_analysis = deepface_analysis[0]
                
                return deepface_analysis.get("dominant_emotion", False)
            except Exception as e:
                print(f"Error in emotion detection: {e}")
                return False