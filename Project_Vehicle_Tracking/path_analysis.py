import numpy as np
import math
import cv2
import os
import json
import numpy as np
import cv2

def is_inside_polygon(point, polygon):
    """
    Check if a point is inside a polygon using OpenCV's pointPolygonTest.

    Args:
    - point: (x, y) coordinates of the point to check.
    - polygon: List of points that define the polygon.

    Returns:
    - True if the point is inside the polygon, False otherwise.
    """
    # Convert the point to a numpy array of type float32 (OpenCV expects this format)
    point = np.array([point], dtype=np.float32)

    # Convert the polygon to a numpy array of type int32 (OpenCV expects this format)
    polygon = np.array(polygon, dtype=np.int32)

    # Use OpenCV's pointPolygonTest to check if the point is inside the polygon
    result = cv2.pointPolygonTest(polygon, tuple(point[0]), False)

    # Return True if the point is inside the polygon (result >= 0), else False
    return result >= 0



def detect_erratic_behavior(path, angle_threshold=30, sample_interval=3):
    """
    Analyzes a subset of points in the path to detect erratic behavior.
    
    Args:
    - path: List of (x, y) points representing the vehicle's path.
    - angle_threshold: Minimum angle (in degrees) to classify behavior as erratic.
    - sample_interval: Interval between sampled points for angle calculation.
    
    Returns:
    - True if erratic behavior is detected, False otherwise.
    """
    if len(path) < sample_interval * 2 + 1:
        return False  # Not enough points for analysis

    # Select points spaced by the sample_interval
    prev_point = np.array(path[-(sample_interval * 2 + 1)])
    mid_point = np.array(path[-(sample_interval + 1)])
    next_point = np.array(path[-1])

    # Calculate vectors and angle between them
    vec1 = mid_point - prev_point
    vec2 = next_point - mid_point
    angle = np.degrees(np.arccos(
        np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
    ))

    if angle < angle_threshold:
        return False  # No erratic behavior
    return True  # Erratic behavior detected


# Function to save the image when erratic behavior is detected
def save_erratic_image(vehicle_id, frame, output_folder, timestamp):
    """
    Save the image when erratic behavior is detected.
    """
    image_filename = f"{output_folder}/erratic_vehicle_{vehicle_id}_{timestamp}.jpg"
    cv2.imwrite(image_filename, frame)
    print(f"Saved erratic image for vehicle {vehicle_id} at {timestamp}")

# Function to track vehicle paths
def track_vehicle_paths(vehicle_ids, vehicle_paths):
    """
    Update paths of detected vehicles.
    """
    for vehicle_id, current_position in vehicle_ids.items():
        if vehicle_id not in vehicle_paths:
            vehicle_paths[vehicle_id] = {'path': [current_position], 'flagged': False, 'saved': False}
        else:
            vehicle_paths[vehicle_id]['path'].append(current_position)
    return vehicle_paths

# Function to monitor the vehicle path and save data when it crosses the bottom region of the homography area
def monitor_and_save_data(vehicle_id, path, frame, save_folder, image_points, vehicle_paths, speed):
    if len(path) > 0 and not vehicle_paths[vehicle_id]['saved']:
        last_position = path[-1]
        inside_region = is_inside_polygon(last_position, image_points)

        # Check if the vehicle has crossed the bottom region
        bottom_region = max(image_points, key=lambda x: x[1])[1]  # Get the bottom y-coordinate of the ROI

        if last_position[1] >= bottom_region:  # If the vehicle crosses the bottom
            # Use the existing save folder from vehicle_paths
            if 'output_p' not in vehicle_paths[vehicle_id]:
                vehicle_paths[vehicle_id]['output_p'] = save_folder

            save_vehicle_path(vehicle_id, path, speed, frame, vehicle_paths[vehicle_id]['output_p'])
            vehicle_paths[vehicle_id]['saved'] = True
            print(f"Vehicle {vehicle_id} path saved at {last_position}")


def save_vehicle_path(vehicle_id, path, speed, frame, output_folder):
    """
    Save the path data and speed information for the vehicle when it crosses the bottom region.
    
    Args:
    - vehicle_id: Unique ID of the vehicle.
    - path: List of coordinates (x, y) representing the vehicle's path.
    - speed: Speed of the vehicle.
    - frame: Current frame (not used in saving data, but can be for additional processing).
    - output_folder: Folder to save the output JSON file.
    
    The function saves the path as a JSON file in the output folder.
    """
    
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Define the filename for the JSON file
    path_filename = os.path.join(output_folder, f"vehicle_{vehicle_id}_path.json")
    
    print(path_filename)
    # Prepare the data to save in JSON format
    vehicle_data = {
        'vehicle_id': vehicle_id,
        'speed': speed,
        'path': [{'x': point[0], 'y': point[1]} for point in path]
    }

    print(vehicle_data)
    
    # Write data to the JSON file
    with open(path_filename, 'w') as json_file:
        json.dump(vehicle_data, json_file, indent=4)
    
    print(f"Saved path and speed data for vehicle {vehicle_id} to {path_filename}")
