import gradio as gr
import os
import numpy as np
import cv2

# from config import config

import mediapipe as mp
from typing import List

from keras_vggface.vggface import VGGFace

# Loading Model for face_vectors
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
print("Model load...")

# Face Detection model
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# db
DETAILED_NPY_FILE = "detailed_data.npy"

def get_face_box(frame) -> List:
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

def put_img_in_db(image, name, age, gender):
    # Placeholder function to process and store data
    if image is None or not name or not age or not gender:
        return "Please provide all details before submitting."
    
    # face_box
    face = get_face_box(image)
    
    if len(face) > 1:
        return {"Message : Multiple faces or Zero face detected"}

    face_vectors = get_face_vector(model, image, face[0])
    
    name = name.replace(" ", "_").lower()
    age = int(age)

    if face_vectors.shape[0] == 1:
        loaded_data = np.load(DETAILED_NPY_FILE, allow_pickle=True).item()
        
        # Ensure the loaded face vectors are 2D
        if len(loaded_data['face_vectors'].shape) == 1:
            loaded_data['face_vectors'] = loaded_data['face_vectors'].reshape(-1, face_vectors.shape[1])

        # Ensure face_vector_i is also 2D
        face_vectors = face_vectors.reshape(1, -1)

        # Append new face vector and label
        appended_face_vectors = np.concatenate((loaded_data['face_vectors'], face_vectors), axis=0)
        appended_labels = np.concatenate((loaded_data['labels'], np.array([name])), axis=0)
        appended_age = np.concatenate((loaded_data['age'], np.array([age])), axis=0)
        appended_gender = np.concatenate((loaded_data['gender'], np.array([gender])), axis=0)

        # Save the updated data
        appended_data = {'face_vectors': appended_face_vectors, 'labels': appended_labels, 'age' : appended_age, 'gender' : appended_gender}
        np.save(DETAILED_NPY_FILE, appended_data)

        print(f"Data added named : {name}")

        return {"Message": "Image and details successfully stored in the database."}
    else:
        return {"Error": "Could not add face to training data"}

def create_gradio_interface():
    with gr.Blocks(title="Data Collection") as interface:
        gr.Markdown("# Capture or Upload Image & Submit Details")
        gr.Markdown("Take your solo picture using webcam or upload it, then enter your details.")
        
        with gr.Tabs():
            with gr.TabItem("Live Capture / Upload"):
                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(sources=["webcam", "upload"], type="numpy", label="Capture or Upload Image")
                        name = gr.Textbox(label="Name")
                        age = gr.Number(label="Age")
                        gender = gr.Dropdown(["Male", "Female", "Other"], label="Gender")
                        submit_btn = gr.Button("Submit")
                    
                    with gr.Column():
                        output_text = gr.Textbox(label="Status", interactive=False)
                
                submit_btn.click(
                    fn=put_img_in_db,
                    inputs=[input_image, name, age, gender],
                    outputs=[output_text]
                )
                
        gr.Markdown(
            """
            ## Instructions:
            1. Capture your solo image using webcam or upload an image.
            2. Enter your Name, Age, and select Gender.
            3. Click 'Submit' to store the details in the database.
            """
        )
    
    return interface

if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch()