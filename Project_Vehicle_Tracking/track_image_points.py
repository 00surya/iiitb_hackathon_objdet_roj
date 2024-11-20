import cv2
import argparse
import os

# Global variables for free selection
points = []

def select_points(event, x, y, flags, param):
    """
    Mouse callback function to select 4 points on the frame.
    """
    global points, frame

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            label_point_position(x, y, frame)
            print(f"Point {len(points)} selected: ({x}, {y})")

def label_point_position(x, y, frame):
    """
    Log the approximate location of the point on the frame.
    """
    height, width, _ = frame.shape

    horizontal_pos = "left" if x < width / 2 else "right"
    vertical_pos = "top" if y < height / 2 else "bottom"
    print(f"The selected point ({x}, {y}) is in the {vertical_pos}-{horizontal_pos} region.")

def main(video_path):
    """
    Main function to read a video, display the first frame,
    and allow the user to select 4 points on the frame.
    """
    global points, frame

    if not os.path.exists(video_path):
        print(f"Error: Video path '{video_path}' does not exist.")
        return

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video.")
        return

    # Read the first frame
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error: Unable to read video frame.")
        return

    # Display the first frame and set up mouse callback
    cv2.namedWindow("Select 4 Points - Press 'q' to confirm")
    cv2.setMouseCallback("Select 4 Points - Press 'q' to confirm", select_points)

    print("Click to select exactly 4 points. Press 'q' to confirm once done.")
    while True:
        temp_frame = frame.copy()

        # Draw points on the frame
        for i, point in enumerate(points):
            cv2.circle(temp_frame, point, 5, (0, 255, 0), -1)
            cv2.putText(temp_frame, f"P{i+1}", point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Select 4 Points - Press 'q' to confirm", temp_frame)
        key = cv2.waitKey(1)

        # Break if 'q' is pressed and 4 points have been selected
        if key == ord('q') and len(points) == 4:
            break

    cv2.destroyAllWindows()

    # Output the selected points
    if len(points) == 4:
        print("Selected Points (in order):")
        for i, point in enumerate(points):
            print(f"Point {i+1}: {point}")
    else:
        print("Error: You must select exactly 4 points.")

if __name__ == "__main__":
    # Argument parser for video path
    parser = argparse.ArgumentParser(description="Select 4 points on a video frame and get coordinates.")
    parser.add_argument("--video_p", type=str, required=True, help="Path to the video file.")
    args = parser.parse_args()

    main(args.video_p)
