import numpy as np
import cv2


def calculate_speed(prev_position, current_position, prev_time, current_time):
    delta_x = current_position[0] - prev_position[0]
    delta_y = current_position[1] - prev_position[1]
    distance_meters = np.sqrt(delta_x ** 2 + delta_y ** 2)
    delta_time = current_time - prev_time

    if delta_time > 0:  # Avoid division by zero
        speed_mps = distance_meters / delta_time
        speed_kmph = speed_mps * 3.6  # Convert to km/h
        return speed_kmph
    return 0


def draw_vehicle_info(frame, vehicle_id, speed, label, bbox, confidence, path, is_erratic):
    x1, y1, x2, y2 = bbox
    # Display vehicle ID, speed, class, and confidence score
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (255, 0, 0)  # Red if erratic, blue otherwise

    thickness = 2

    # First line of text (Vehicle ID and Speed)
    cv2.putText(frame, f'ID: {vehicle_id}', (x2, y1-80 ), font, font_scale, color, thickness)
    cv2.putText(frame, f'Speed: {speed:.2f} km/h', (x2, y1 - 60), font, font_scale, color, thickness)

    # Second line of text (Class and Confidence)
    cv2.putText(frame, f'Class: {label}', (x2, y1 - 40), font, font_scale, color, thickness)
    cv2.putText(frame, f'Conf: {confidence:.2f}', (x2, y1 - 20), font, font_scale, color, thickness)

    # Draw the vehicle path (last 10 points)
    for i in range(1, len(path)):
        cv2.line(frame, path[i - 1], path[i], (0, 255, 0), 2)

    # Draw bounding box
    box_color = (0, 0, 255) if is_erratic else color
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    return frame


def draw_homography_square(frame, transformed_box):
    transformed_box = transformed_box.reshape((-1, 2))  # Reshape to (N, 2) for polygon drawing
    cv2.polylines(frame, [transformed_box], isClosed=True, color=(255, 0, 0), thickness=2)


def compute_homography(real_world_points, image_points):
    homography_matrix, _ = cv2.findHomography(real_world_points, image_points)
    return homography_matrix

def transform_to_real_world(image_points, homography_matrix):
    transformed_points = cv2.perspectiveTransform(image_points.reshape(-1, 1, 2), homography_matrix)
    return np.int32(transformed_points)
