
import cv2
import numpy as np
import time
import os
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

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


def get_face_vector(model, frame, points):
    try:
        crop_face = crop_image(frame, points)
        crop_face = np.expand_dims(crop_face, axis=0)

        face_vector = model.predict(
            crop_face
        )

        return face_vector
    except Exception as e:
        raise ValueError("Error in Face Vector : ", e)

def identify_user(face_vector, trained_face_vectors):
    try:
        # crop_face = crop_image(frame, points)
        # crop_face = np.expand_dims(crop_face, axis=0)

        # face_vector = model.predict(
        #     crop_face
        # )

        # face_vector = get_face_vector(model, frame, points)

        face_scores = cosine_similarity(face_vector, trained_face_vectors)
        max_prob = np.max(face_scores)
        index = np.argmax(face_scores)

        return max_prob, index
    except Exception as e:
        print("Error in identify_users function : ", e)
        return 0, 0
    
