import numpy as np
import time
from collections import deque

class ShotPredictionAI:
    """
    Lightweight, local Shot Prediction AI.
    Uses a simple K-Nearest Neighbors (KNN) classifier implemented in pure Python/NumPy
    to predict shot direction (LEFT, CENTER, RIGHT) from kick features.
    """
    def __init__(self, k=3):
        self.k = k
        # Features: [kick_angle, ankle_velocity, body_orientation]
        # Labels: 0 = LEFT, 1 = CENTER, 2 = RIGHT
        self.X_train = []
        self.y_train = []
        # Prepopulate with some intuitive default training examples
        # Angle (deg relative to vertical), Velocity (normalized), Body Orient (deg)
        default_samples = [
            # Left shots
            ([15.0, 0.15, -5.0], 0),
            ([25.0, 0.20, -10.0], 0),
            # Center shots
            ([0.0, 0.12, 0.0], 1),
            ([2.0, 0.18, 1.0], 1),
            # Right shots
            ([-15.0, 0.15, 5.0], 2),
            ([-25.0, 0.20, 10.0], 2)
        ]
        for features, label in default_samples:
            self.X_train.append(features)
            self.y_train.append(label)

    def train_sample(self, kick_angle, ankle_velocity, body_orientation, label):
        """
        Record a new sample to train the classifier online.
        """
        self.X_train.append([float(kick_angle), float(ankle_velocity), float(body_orientation)])
        self.y_train.append(int(label))
        
        # Keep training set bound to last 100 shots to adapt to current player
        if len(self.X_train) > 100:
            self.X_train.pop(0)
            self.y_train.pop(0)

    def predict(self, kick_angle, ankle_velocity, body_orientation):
        """
        Predict the shot direction based on input features.
        Returns predicted label (0, 1, 2) and confidence value.
        """
        if not self.X_train:
            return 1, 1.0 # Fallback to CENTER
            
        x = np.array([kick_angle, ankle_velocity, body_orientation])
        X = np.array(self.X_train)
        
        # Normalize features dynamically (Z-score normalization to avoid scale dominance)
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1e-5 # avoid division by zero
        
        X_norm = (X - mean) / std
        x_norm = (x - mean) / std
        
        # Calculate Euclidean distances
        distances = np.linalg.norm(X_norm - x_norm, axis=1)
        
        # Get k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_labels = [self.y_train[i] for i in k_indices]
        
        # Vote
        counts = np.bincount(k_labels, minlength=3)
        prediction = int(np.argmax(counts))
        confidence = float(counts[prediction] / self.k)
        
        return prediction, confidence


class KickDetector:
    """
    Tracks ankles and body joints to detect kicks, calculate velocity,
    determine power and shot properties, and predict aiming direction.
    """
    def __init__(self, history_len=5, speed_threshold=0.04, cool_down_time=1.5):
        self.history_len = history_len
        self.default_speed_threshold = speed_threshold # normalized units per frame
        self.speed_threshold = speed_threshold
        self.cool_down_time = cool_down_time
        
        # History buffers for ankle positions: deque of (timestamp, x, y)
        self.left_ankle_history = deque(maxlen=history_len)
        self.right_ankle_history = deque(maxlen=history_len)
        
        self.last_kick_time = 0
        self.prediction_ai = ShotPredictionAI()
        
        # Calibration factors
        self.shoulder_width_ref = 0.15 # default normalized shoulder width
        self.leg_length_ref = 0.4 # default normalized leg length

    def update_calibration(self, shoulder_width, leg_length):
        """
        Adjust thresholds based on player size calibration.
        """
        if shoulder_width > 0:
            self.shoulder_width_ref = shoulder_width
        if leg_length > 0:
            self.leg_length_ref = leg_length
            
        # Dynamically scale sensitivity: smaller/closer players have larger movements
        # We adjust speed threshold relative to leg length.
        # If leg is longer (player closer), they move more pixels, so increase threshold.
        self.speed_threshold = self.default_speed_threshold * (self.leg_length_ref / 0.4)
        # Clamp to reasonable values
        self.speed_threshold = max(0.02, min(0.08, self.speed_threshold))

    def _calculate_velocity(self, history):
        """
        Calculates instantaneous velocity from ankle history.
        Velocity is computed as displacement over time interval.
        """
        if len(history) < 2:
            return 0.0
            
        # Compute distance between oldest and newest point in the history
        t_old, x_old, y_old = history[0]
        t_new, x_new, y_new = history[-1]
        
        dt = t_new - t_old
        if dt <= 0:
            return 0.0
            
        dist = np.sqrt((x_new - x_old)**2 + (y_new - y_old)**2)
        # Normalize velocity by time to make it frame-rate independent
        return dist / dt

    def process_landmarks(self, joints):
        """
        Processes key joint coordinates and checks for a kick.
        joints: dict of joint names to coordinate dicts
        Returns a dict of event details if a kick is triggered, else None.
        """
        now = time.time()
        
        if 'left_ankle' not in joints or 'right_ankle' not in joints:
            return None
            
        # Extract normalized ankle coordinates
        la_x, la_y = joints['left_ankle']['x_norm'], joints['left_ankle']['y_norm']
        ra_x, ra_y = joints['right_ankle']['x_norm'], joints['right_ankle']['y_norm']
        
        self.left_ankle_history.append((now, la_x, la_y))
        self.right_ankle_history.append((now, ra_x, ra_y))
        
        # If cooled down, compute velocities
        if now - self.last_kick_time < self.cool_down_time:
            return None
            
        left_vel = self._calculate_velocity(self.left_ankle_history)
        right_vel = self._calculate_velocity(self.right_ankle_history)
        
        kick_triggered = False
        kicking_leg = None
        peak_velocity = 0.0
        
        # Check if either velocity exceeds threshold
        if left_vel > self.speed_threshold and left_vel > right_vel:
            kick_triggered = True
            kicking_leg = "left"
            peak_velocity = left_vel
        elif right_vel > self.speed_threshold:
            kick_triggered = True
            kicking_leg = "right"
            peak_velocity = right_vel
            
        if kick_triggered:
            self.last_kick_time = now
            return self._create_kick_event(joints, kicking_leg, peak_velocity)
            
        return None

    def _create_kick_event(self, joints, leg, velocity):
        """
        Builds the detailed parameters of the kick event.
        """
        # 1. Determine body center (average of hips)
        lh_x = joints['left_hip']['x_norm']
        rh_x = joints['right_hip']['x_norm']
        body_center_x = (lh_x + rh_x) / 2.0
        
        # 2. Get kicking ankle position
        ankle_key = f"{leg}_ankle"
        knee_key = f"{leg}_knee"
        ankle_x = joints[ankle_key]['x_norm']
        ankle_y = joints[ankle_key]['y_norm']
        knee_y = joints[knee_key]['y_norm']
        
        # 3. Determine shot direction
        # We look at the ankle's horizontal offset from body center.
        # Since camera is mirrored, moving leg to the left of the image (smaller x) means
        # kicking left. Moving leg to the right of the image (larger x) means kicking right.
        # Use shoulder width to scale the center zone margin.
        margin = self.shoulder_width_ref * 0.4
        
        if ankle_x < body_center_x - margin:
            direction = "LEFT"
            dir_label = 0
        elif ankle_x > body_center_x + margin:
            direction = "RIGHT"
            dir_label = 2
        else:
            direction = "CENTER"
            dir_label = 1
            
        # 4. Determine shot height (LOW, MID, HIGH) based on vertical kick angle/height
        # Compare ankle height to knee height
        # Smaller y means higher up.
        height_diff = knee_y - ankle_y # if ankle is higher than knee, height_diff is positive
        
        # Thresholds relative to leg length
        high_threshold = self.leg_length_ref * 0.1
        low_threshold = -self.leg_length_ref * 0.1
        
        if height_diff > high_threshold:
            height = "HIGH"
        elif height_diff < low_threshold:
            height = "LOW"
        else:
            height = "MID"
            
        # 5. Determine shot power (0 - 100%)
        # Map velocity to power. The velocity usually ranges from speed_threshold to roughly 3 * speed_threshold.
        min_vel = self.speed_threshold
        max_vel = self.speed_threshold * 2.5
        power = int(np.clip(((velocity - min_vel) / (max_vel - min_vel)) * 100, 15, 100))
        
        # 6. Calculate Kick Angle & Body Orientation for AI Shot Prediction
        # Kick angle is the angle of the kicking leg (knee to ankle) relative to vertical
        knee_x = joints[knee_key]['x_norm']
        dx = ankle_x - knee_x
        dy = ankle_y - knee_y
        kick_angle = np.degrees(np.arctan2(dx, dy if dy != 0 else 0.001))
        
        # Body orientation can be estimated using shoulder tilt
        ls_y = joints['left_shoulder']['y_norm']
        rs_y = joints['right_shoulder']['y_norm']
        ls_x = joints['left_shoulder']['x_norm']
        rs_x = joints['right_shoulder']['x_norm']
        body_orientation = np.degrees(np.arctan2(ls_y - rs_y, ls_x - rs_x if ls_x != rs_x else 0.001))
        
        # Make a prediction using the AI
        predicted_dir_label, confidence = self.prediction_ai.predict(kick_angle, velocity, body_orientation)
        direction_names = ["LEFT", "CENTER", "RIGHT"]
        predicted_direction = direction_names[predicted_dir_label]
        
        # Train the model online with this outcome (actual direction determined by the geometry)
        self.prediction_ai.train_sample(kick_angle, velocity, body_orientation, dir_label)
        
        return {
            'leg': leg,
            'velocity': velocity,
            'power': power,
            'direction': direction,
            'height': height,
            'kick_angle': kick_angle,
            'body_orientation': body_orientation,
            'prediction': predicted_direction,
            'confidence': confidence,
            'timestamp': time.time()
        }
