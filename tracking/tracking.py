from norfair import Tracker, Detection
import numpy as np

def create_tracker():
    """Creates and returns a Norfair tracker."""
    return Tracker(distance_function="euclidean", distance_threshold=35)

def track_objects(detections, tracker):
    """Tracks objects using Norfair"""
    norfair_detections = []
    for x, y, w, h in detections:
        centroid = np.array([(x + w) / 2, (y + h) / 2])
        norfair_detections.append(Detection(centroid, data={"bbox": (x, y, w, h)}))
    return tracker.update(norfair_detections)
