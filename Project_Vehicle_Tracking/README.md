
# Vehicle Tracking and Behavior Analysis System

## Project Overview

This project aims to track and analyze vehicles in high-speed highway videos. By leveraging the **YOLOv8** model for vehicle detection and a **Kalman filter** for vehicle tracking, we achieve precise tracking and behavior analysis. The primary goal is to assign unique IDs to vehicles, predict their future positions using the Kalman filter, and analyze their behavior based on their path, speed, and erratic movements.

![Demo Gif](sprites/tracking.gif)

### Key Features:

- **Vehicle Tracking with Kalman Filter**:  
  The system tracks vehicles across video frames, utilizing a Kalman filter to predict the future location of each vehicle based on its previous state (position and velocity).  
  **ID Assignment**: By comparing the predicted and actual positions, the system assigns accurate IDs to vehicles, even when there is temporary occlusion or loss of tracking.

- **Frame Skipping for Robust Tracking**:  
  To enhance the robustness of tracking, we incorporated frame skipping. This means that even if a vehicle is temporarily lost in some frames, the Kalman filter's prediction helps re-associate the vehicle's ID with its correct trajectory in the next frames.

- **Speed and Behavior Analysis**:  
  Vehicles traveling at high speeds (greater than 80 km/h) are tracked and analyzed for their behavior. This system calculates the speed of vehicles based on their real-world coordinates (using a homography matrix) and stores the path data for further analysis.

- **Erratic Behavior Detection**:  
  The system can detect erratic behavior based on the vehicle's movement path. If the vehicle takes sharp turns or deviates from a typical path (calculated by the angle between consecutive points), it flags the vehicle for erratic driving.

---

## How Kalman Filter is Used for Vehicle Tracking

In this system, the **Kalman filter** is employed for robust vehicle tracking. The filter helps predict the future position of a vehicle based on its previous position and velocity, ensuring that the ID of the vehicle remains consistent even when the vehicle is temporarily out of the camera's view or occluded by other vehicles.

### Kalman Filter Steps:

1. **Prediction Step**:  
   The Kalman filter uses the previous state of the vehicle (position and velocity) to predict its future state (next position).  
   The state vector `[x, y, dx, dy]` is used, where:  
   - `(x, y)` is the position of the vehicle,  
   - `(dx, dy)` is the velocity.

2. **Update Step**:  
   When the vehicle reappears in the frame, its actual position is measured, and the Kalman filter updates its prediction.  
   The predicted position and the measured position are fused to give a more accurate estimate of the vehicle's location.

3. **Frame Skipping**:  
   In some cases, vehicles may be temporarily lost. Frame skipping allows us to predict the vehicle's location even if we miss a few frames, making the ID assignment more robust.

### ID Assignment and Tracking:
- The predicted position is compared to the detected vehicle’s position in the current frame. If the predicted and detected positions match closely, the same ID is assigned to the vehicle.  
  This helps in continuously tracking each vehicle, even if there is some temporary loss of detection.

---

## Path Analysis and Speed Calculation

This project assumes that the vehicles are traveling at high speeds (greater than 80 km/h), which is typical for highways. The **path analysis** module helps track the trajectory of each vehicle and analyze its speed and behavior.

### Key Steps:
1. **Real-World Speed Calculation**:  
   The speed of a vehicle is calculated when it moves from one point to another in real-world coordinates (derived from the homography matrix). The calculation is based on the Euclidean distance between two consecutive points in real-world coordinates and the time taken to traverse that distance.

2. **Homography Matrix**:  
   The system uses a **homography matrix** to transform coordinates from the image frame to real-world coordinates. This transformation is essential for accurately estimating speed and detecting erratic behavior.  
   ![Homography Matrix](sprites/homgraphy.png)

   The homography matrix is calculated using real-world coordinates and corresponding points in the image frame.

---

## Setting Up the Real-World Coordinate System

To calibrate the system, you need to define a real-world region (such as a 10x7 meter square) and map it to the image coordinates. This is the first step after setting up the environment.

### Steps for Calibration:

1. **Define the Real-World Coordinates**:  
   Specify the dimensions of the real-world area you want to map (e.g., 10x7 meters).  
   Identify the corresponding points in the image that represent these real-world coordinates.

2. **Run the Calibration Script**:  
   The `compute_homography` function calculates the homography matrix based on the real-world and image points.  
   The transformation is used to map the vehicle's detected coordinates to the real-world coordinates.

3. **Example**:  
   Suppose the real-world area is a 10x7 meter square. You will select the four corners of this square in the video frame and input these points into the calibration script.  
   The system will then compute the homography matrix that transforms these image points to real-world coordinates.

---

## Visualization and Tracking Flow

The vehicle tracking and behavior analysis are visualized in the following steps:

1. **Detection**:  
   YOLOv8 is used to detect vehicles in each frame. These detections are processed to extract vehicle positions and bounding boxes.  
  

2. **Tracking**:  
   The Kalman filter is used to track the vehicles and predict their future positions.  
   IDs are assigned to vehicles, ensuring they remain consistent across frames.

3. **Speed and Path**:  
   The speed and path of each vehicle are calculated and displayed on the frame.  
   The path is drawn by connecting the positions of the vehicle in previous frames.  
  

4. **Erratic Behavior**:  
   The system analyzes the path of each vehicle. If erratic behavior (sharp turns or abrupt movements) is detected, the vehicle is flagged.  
   The behavior is visualized by drawing the bounding box in a different color and saving the frame if necessary.  
   

5. **Final Visualization**:  
   The final frame is displayed with annotations such as vehicle ID, speed, class (car, truck, etc.), and confidence score.  
   The vehicle’s path is also drawn, and erratic behavior is flagged visually.

---

## Fine-Tuning the YOLOv8 Model

The system utilizes a **fine-tuned YOLOv8** model, trained on a custom dataset (XYZ dataset). Fine-tuning improves the model's accuracy for vehicle detection in the context of high-speed highway videos.

### Training Process:

1. **Dataset**: The XYZ dataset (to be provided) contains labeled images of vehicles.
2. **Fine-Tuning**: The YOLOv8 model is trained on this dataset to improve its performance in detecting vehicles in highway scenarios.

---

## Script Workflow

The overall flow of the scripts is as follows:

1. **Setup**:  
   Install dependencies using the `setup.sh` script or manually.  
   Calibrate the real-world coordinates by selecting points in the video frame.

2. **Inference**:  
   The `inference.py` script processes the video, detecting and tracking vehicles.  
   It uses YOLOv8 for detection, the Kalman filter for tracking, and computes the speed and path for each vehicle.  
   Erratic behavior is detected, and the system flags and saves images of such events.

3. **Path Analysis**:  
   The `path_analysis.py` script detects erratic behavior and saves data about vehicle paths when they cross a defined region.

4. **Vehicle Tracking**:  
   The `tracking.py` script assigns unique IDs to vehicles and tracks their movement using the Kalman filter.

---

## Setup Instructions

Now, let's set up the environment and run the project.

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-repository/vehicle-tracking-system.git
   cd vehicle-tracking-system
