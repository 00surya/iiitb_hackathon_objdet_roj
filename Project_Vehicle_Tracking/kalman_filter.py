import numpy as np

# Kalman Tracker class for object tracking
class KalmanTracker:
    def __init__(self):
        self.state = np.array([0, 0, 0, 0], dtype=float)  # [x, y, dx, dy]
        self.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])  # State transition matrix
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])  # Observation matrix
        self.P = np.eye(4) * 1000  # Covariance matrix
        self.Q = np.eye(4)  # Process noise
        self.R = np.eye(2) * 5  # Measurement noise
        self.missed_frames = 0
        self.id = None  # Tracker ID

    def predict(self):
        self.state = np.dot(self.F, self.state)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.state[:2]

    def update(self, measurement):
        y = measurement - np.dot(self.H, self.state)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.state = self.state + np.dot(K, y)
        self.P = np.dot(np.eye(4) - np.dot(K, self.H), self.P)
        self.missed_frames = 0  # Reset missed frames when updated

    def increment_missed_frames(self):
        self.missed_frames += 1
