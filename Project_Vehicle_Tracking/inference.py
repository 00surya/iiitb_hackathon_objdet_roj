import cv2
import numpy as np
import yaml
import os
import time
from tracking import assign_ids_with_kalman, track_vehicle_paths
from utils import draw_vehicle_info, transform_to_real_world, draw_homography_square, compute_homography
from ultralytics import YOLO
from path_analysis import detect_erratic_behavior, save_erratic_image, monitor_and_save_data

# Global variables to store the YOLO model and class names
model = None
class_names = None

# Function to initialize the YOLO model
def init_model(model_p):
    """
    Load the YOLO model and initialize the class names.
    
    Args:
        model_p (str): Path to the YOLO model weights.
    """
    global model, class_names
    model = YOLO(model_p)  # Load YOLO model from specified path
    class_names = model.names  # Retrieve class names
    return 

# Function to process each video frame
def process_frame(frame, results, previous_ids, vehicle_paths, trackers, fps, native_fps, image_points):
    """
    Process a single frame for vehicle detection, tracking, and analysis.
    
    Args:
        frame (ndarray): The current video frame.
        results (list): Detection results from YOLO.
        previous_ids (dict): Previously tracked vehicle IDs.
        vehicle_paths (dict): Dictionary storing paths and speed for each vehicle.
        trackers (dict): Kalman trackers for smoothing detections.
        fps (float): Frames per second of the current video stream.
        native_fps (float): Native FPS of the video file.
        image_points (ndarray): Points defining the region of interest in the image.
    
    Returns:
        tuple: Annotated frame, updated trackers, and updated vehicle paths.
    """
    delta_time = 1 / native_fps  # Time per frame in seconds

    # Assign IDs to detected objects and update Kalman trackers
    new_ids, trackers = assign_ids_with_kalman(results, trackers)

    # Update vehicle paths and calculate speed
    vehicle_paths = track_vehicle_paths(new_ids, vehicle_paths, delta_time=delta_time)

    for vehicle_id, center in new_ids.items():
        for result in results[0].boxes:
            class_idx = int(result.cls[0])  # Class index of the detected object
            label = class_names[class_idx]  # Class label (e.g., "car")
            confidence = result.conf[0]  # Confidence score of the detection
            x1, y1, x2, y2 = map(int, result.xyxy[0])  # Bounding box coordinates
            bbox_center = ((x1 + x2) // 2, (y1 + y2) // 2)  # Bounding box center

            # Match detected center with tracked center
            if np.linalg.norm(np.array(bbox_center) - np.array(center)) < 50:
                speed = vehicle_paths[vehicle_id]['speed']  # Vehicle speed
                path = vehicle_paths[vehicle_id]['path']  # Vehicle path

                # Detect erratic behavior based on the path
                is_erratic = detect_erratic_behavior(path, angle_threshold=30)

                # Create output directory for erratic behavior
                timestamp = int(time.time() * 1000)  # Current timestamp in milliseconds

                try:
                    output_p = vehicle_paths[vehicle_id]['output_p']
                except:
                    output_p = os.path.join("catches", str(timestamp))
                    vehicle_paths[vehicle_id]['output_p'] = output_p

                # Save image if erratic behavior is detected
                if is_erratic and not vehicle_paths[vehicle_id]['flagged']:
                    os.makedirs(output_p, exist_ok=True)
                    save_erratic_image(vehicle_id, frame, output_p, timestamp)
                    vehicle_paths[vehicle_id]['flagged'] = True
                    vehicle_paths[vehicle_id]['output_p'] = output_p

                
                # Monitor flagged vehicles and save their paths
                if vehicle_paths[vehicle_id]['flagged']:
                    vehicle_paths[vehicle_id]['saved'] = False
                    monitor_and_save_data(vehicle_id, path, frame, vehicle_paths[vehicle_id]['output_p'], image_points, vehicle_paths, speed)

                # Annotate frame with vehicle info
                frame = draw_vehicle_info(frame, vehicle_id, speed, label, (x1, y1, x2, y2), confidence, path, is_erratic)

    # Display FPS on the frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    return frame, trackers, vehicle_paths

# Function to read points from a YAML file
def read_points_from_yaml(file_path):
    """
    Read calibration points (real-world and image) from a YAML file.
    
    Args:
        file_path (str): Path to the YAML file.
    
    Returns:
        tuple: Arrays of real-world and image points.
    """
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    real_world_points_np = np.array(data['real_world_points'], dtype=np.float32)
    image_points_np = np.array(data['image_points'], dtype=np.float32)

    return real_world_points_np, image_points_np

# Main inference function
def infer(video_path, output_path=None, device='cpu', model_p="models/m1/weights/best.pt", calibration_points='data/points/calibration_points.yml'):
    """
    Perform inference on a video using YOLO and analyze vehicle behavior.
    
    Args:
        video_path (str): Path to the input video file.
        output_path (str): Path to save output results (optional).
        device (str): Device to run inference ('cpu' or 'cuda').
        model_p (str): Path to the YOLO model weights.
        calibration_points (str): Path to the YAML file with calibration points.
    """
    # Initialize YOLO model
    init_model(model_p=model_p)
    
    cap = cv2.VideoCapture(video_path)  # Open video file
    prev_time = time.time()
    vehicle_paths = {}  # Vehicle paths and data
    previous_ids = {}  # Previous IDs for tracking
    trackers = {}  # Kalman trackers for objects

    # Read real-world and image points from YAML file
    real_world_points, image_points = read_points_from_yaml(calibration_points)

    # Compute homography matrix and transform points
    homography_matrix = compute_homography(real_world_points, image_points)
    transformed_box = transform_to_real_world(real_world_points, homography_matrix)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Perform YOLO inference
        results = model(frame, device=device)

        # Process the frame for tracking and analysis
        annotated_frame, previous_ids, vehicle_paths = process_frame(
            frame, results, previous_ids, vehicle_paths, trackers, fps, cap.get(cv2.CAP_PROP_FPS), image_points)

        # Draw the homography square on the frame
        draw_homography_square(annotated_frame, transformed_box)

        # Display the frame
        cv2.imshow('Frame', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
