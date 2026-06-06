import cv2
import mediapipe as mp
import numpy as np

class PoseDetector:
    """
    Skeletal pose detector using MediaPipe Pose.
    Tracks hips, knees, ankles, shoulders and provides coordinates.
    """
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.results = None

    def find_pose(self, frame, draw=True):
        """
        Process the OpenCV BGR frame and optionally draw the skeleton overlay.
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(frame_rgb)
        
        if self.results.pose_landmarks and draw:
            # Draw standard skeleton overlay
            self.mp_draw.draw_landmarks(
                frame,
                self.results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        return frame

    def get_landmarks(self, frame_shape):
        """
        Extract and return pose landmarks in normalized and pixel coordinates.
        frame_shape: (height, width, channels)
        Returns a dictionary mapping landmark indices to (x_pixel, y_pixel, x_norm, y_norm, z_norm, visibility).
        """
        landmarks = {}
        if self.results and self.results.pose_landmarks:
            h, w = frame_shape[0], frame_shape[1]
            for idx, lm in enumerate(self.results.pose_landmarks.landmark):
                # Ignore landmarks with extremely low visibility/confidence
                px_x = int(lm.x * w)
                px_y = int(lm.y * h)
                landmarks[idx] = {
                    'x': px_x,
                    'y': px_y,
                    'x_norm': lm.x,
                    'y_norm': lm.y,
                    'z_norm': lm.z,
                    'visibility': lm.visibility
                }
        return landmarks

    def get_key_joints(self, landmarks):
        """
        Convenience method to retrieve critical landmarks.
        Returns a dict of key joint names to coordinates.
        """
        joints = {}
        if not landmarks:
            return joints
            
        # Mapping key anatomical features to MediaPipe indices
        mapping = {
            'left_hip': self.mp_pose.PoseLandmark.LEFT_HIP.value,
            'right_hip': self.mp_pose.PoseLandmark.RIGHT_HIP.value,
            'left_knee': self.mp_pose.PoseLandmark.LEFT_KNEE.value,
            'right_knee': self.mp_pose.PoseLandmark.RIGHT_KNEE.value,
            'left_ankle': self.mp_pose.PoseLandmark.LEFT_ANKLE.value,
            'right_ankle': self.mp_pose.PoseLandmark.RIGHT_ANKLE.value,
            'left_shoulder': self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            'right_shoulder': self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            'left_wrist': self.mp_pose.PoseLandmark.LEFT_WRIST.value,
            'right_wrist': self.mp_pose.PoseLandmark.RIGHT_WRIST.value
        }
        
        for name, idx in mapping.items():
            if idx in landmarks:
                joints[name] = landmarks[idx]
        return joints
