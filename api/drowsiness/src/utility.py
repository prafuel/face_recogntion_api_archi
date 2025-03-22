
import time
from scipy.spatial import distance
import numpy as np


BLINK_PATTERN_THRESH = 5  # Number of blinks to check for pattern

# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    # Compute the euclidean distance between vertical eye landmarks
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    # Compute the euclidean distance between horizontal eye landmarks
    C = distance.euclidean(eye[0], eye[3])
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

# Function to calculate mouth aspect ratio (MAR)
def mouth_aspect_ratio(mouth):
    # Compute vertical distances
    A = distance.euclidean(mouth[2], mouth[10])  # Upper and lower lip
    B = distance.euclidean(mouth[4], mouth[8])   # Upper and lower lip
    # Compute horizontal distance
    C = distance.euclidean(mouth[0], mouth[6])   # Mouth width
    # Compute mouth aspect ratio
    mar = (A + B) / (2.0 * C)
    return mar

# Function to detect head movement
def head_movement(prev_landmarks, curr_landmarks):
    if prev_landmarks is None or curr_landmarks is None:
        return 0
    # Calculate average movement of all landmarks
    movement = np.mean(np.sqrt(np.sum((prev_landmarks - curr_landmarks) ** 2, axis=1)))
    return movement

# Function to analyze blink patterns
def analyze_blink_pattern(pattern, time_since_last_blink):
    if len(pattern) < BLINK_PATTERN_THRESH:
        return "Normal"
    
    # Rapid blinking detection
    if np.mean(pattern) < 0.4:  # If average time between blinks is low
        return "Rapid Blinking - Possible fatigue or eye strain"
    
    # Infrequent blinking detection
    if time_since_last_blink > 7.0:  # If no blink for more than 7 seconds
        return "Infrequent Blinking - Possible intense concentration or dry eyes"
    
    return "Normal"

# Function to run in a separate thread for processing concentration
def concentration_analyzer():
    global blink_pattern, last_blink_time
    while True:
        time.sleep(1)
        # Analyze time since last blink
        time_since_last_blink = time.time() - last_blink_time
        
        # Get concentration status based on blink pattern
        concentration_status = analyze_blink_pattern(blink_pattern, time_since_last_blink)
        print(f"Concentration Status: {concentration_status}")
        
        # Limit the pattern list size
        if len(blink_pattern) > 20:
            blink_pattern.pop(0)


def get_kernel(height, width):
    kernel_size = ((height + 15) // 16, (width + 15) // 16)
    return kernel_size


def get_width_height(patch_info):
    w_input = int(patch_info.split('x')[-1])
    h_input = int(patch_info.split('x')[0].split('_')[-1])
    return w_input,h_input