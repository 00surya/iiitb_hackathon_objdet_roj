import numpy as np
from utils import calculate_speed
import time
from collections import deque, defaultdict
from scipy.optimize import linear_sum_assignment
from kalman_filter import KalmanTracker

# Use a defaultdict to manage Kalman trackers for all vehicles
trackers = defaultdict(KalmanTracker)

# Speed history deque to store recent speed values for smoothing
speed_history = deque(maxlen=10)

# Function to assign IDs to detected objects using Kalman trackers
def assign_ids_with_kalman(results, trackers):
    """
    Assign unique IDs to detected objects using Kalman filter-based trackers.

    Args:
        results (list): YOLO detection results containing bounding boxes and classes.
        trackers (defaultdict): A dictionary of active Kalman trackers.

    Returns:
        tuple: A dictionary of assigned IDs and their centers, updated trackers.
    """
    new_ids = {}  # Dictionary to store assigned IDs and corresponding centers
    detections = []  # List to store current frame detections

    # Extract centers of detected bounding boxes
    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])  # Bounding box coordinates
        center = ((x1 + x2) // 2, (y1 + y2) // 2)  # Calculate center
        detections.append(center)

    if len(detections) > 0:
        # Create a cost matrix based on Euclidean distance between trackers and detections
        cost_matrix = np.zeros((len(trackers), len(detections)), dtype=np.float32)

        for i, (tracker_id, tracker) in enumerate(trackers.items()):
            predicted_center = tracker.predict()  # Predict next position of tracker
            for j, detection in enumerate(detections):
                cost_matrix[i, j] = np.linalg.norm(np.array(predicted_center) - np.array(detection))

        # Solve the assignment problem using the Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Update trackers with assigned detections
        assigned_ids = {}
        for r, c in zip(row_ind, col_ind):
            tracker_id = list(trackers.keys())[r]
            trackers[tracker_id].update(detections[c])  # Update tracker with detection
            assigned_ids[tracker_id] = detections[c]  # Assign ID to detection

        # Create new trackers for unassigned detections
        for i, detection in enumerate(detections):
            if i not in col_ind:
                new_tracker = KalmanTracker()  # Initialize a new tracker
                new_tracker.update(detection)  # Update tracker with detection
                new_tracker.id = len(trackers) + 1  # Assign a unique ID
                trackers[new_tracker.id] = new_tracker  # Add new tracker to the dictionary
                assigned_ids[new_tracker.id] = detection

        return assigned_ids, trackers
    return {}, trackers


# Function to track vehicle paths and update their states
def track_vehicle_paths(new_ids, vehicle_paths, max_age=5, delta_time=1.0):
    """
    Track the paths of vehicles and update their speeds.

    Args:
        new_ids (dict): Dictionary of assigned IDs and their current centers.
        vehicle_paths (dict): Dictionary storing paths, speeds, and other metadata for vehicles.
        max_age (int): Maximum age (in frames) before a tracker is removed.
        delta_time (float): Time interval between frames for speed calculation.

    Returns:
        dict: Updated vehicle paths.
    """
    to_remove = []  # List of vehicle IDs to be removed

    # Update paths for existing trackers or create new entries
    for vehicle_id, center in new_ids.items():
        if vehicle_id not in vehicle_paths:
            # Initialize a new entry for the vehicle
            vehicle_paths[vehicle_id] = {
                'path': [],  # List to store the path of the vehicle
                'age': 0,  # Age counter for missed detections
                'last_position': center,  # Last known position of the vehicle
                'speed': 0,  # Current speed of the vehicle
                'previous_time': 0,  # Timestamp of the last update
                'flagged': False  # Whether the vehicle is flagged for erratic behavior
            }

        # Append current center to the vehicle's path
        vehicle_paths[vehicle_id]['path'].append(center)
        vehicle_paths[vehicle_id]['age'] = 0  # Reset the age (tracker is detected)

        prev_position = vehicle_paths[vehicle_id]['last_position']
        prev_time = vehicle_paths[vehicle_id]['previous_time']
        current_time = time.time()  # Get the current timestamp

        # Calculate the vehicle's speed in km/h
        speed_kmph = calculate_speed(prev_position, center, prev_time, current_time)

        # Add the current speed to the history deque
        speed_history.append(speed_kmph)

        # Smooth the speed using the moving average of the speed history
        smoothed_speed = np.mean(speed_history)

        # Update the vehicle's metadata
        vehicle_paths[vehicle_id]['speed'] = smoothed_speed
        vehicle_paths[vehicle_id]['previous_time'] = current_time
        vehicle_paths[vehicle_id]['last_position'] = center

    # Check for trackers that have exceeded the maximum age
    for vehicle_id, data in vehicle_paths.items():
        if vehicle_id not in new_ids:
            # Increment the age of trackers that are not detected
            vehicle_paths[vehicle_id]['age'] += 1
            trackers[vehicle_id].increment_missed_frames()  # Update missed frame count in tracker

            if vehicle_paths[vehicle_id]['age'] > max_age or trackers[vehicle_id].missed_frames > max_age:
                to_remove.append(vehicle_id)  # Mark tracker for removal

    # Remove outdated trackers and their paths
    for vehicle_id in to_remove:
        del vehicle_paths[vehicle_id]
        del trackers[vehicle_id]

    return vehicle_paths
