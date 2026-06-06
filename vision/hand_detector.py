import cv2
import mediapipe as mp

class HandDetector:
    """
    Hand detector using MediaPipe Hands.
    Used in Goalkeeper mode to track player hand locations.
    """
    def __init__(self, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.results = None

    def find_hands(self, frame, draw=True):
        """
        Process the OpenCV BGR frame and optionally draw hand landmarks.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frame_rgb)
        
        if self.results.multi_hand_landmarks and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        return frame

    def get_hand_positions(self, frame_shape):
        """
        Retrieves hand coordinates from current tracking results.
        Returns a list of dictionaries, one for each hand found:
        [
          {
            'type': 'Left' or 'Right', (MediaPipe label)
            'wrist': (x, y),
            'palm_center': (x, y),
            'landmarks': {idx: (x, y, x_norm, y_norm)}
          },
          ...
        ]
        """
        hands_data = []
        if not self.results or not self.results.multi_hand_landmarks:
            return hands_data
            
        h, w = frame_shape[0], frame_shape[1]
        
        # Get hand labels (Left/Right)
        # Note: MediaPipe hand labeling is inverted compared to what you see on screen,
        # but we can obtain the handedness metadata.
        handedness = []
        if self.results.multi_handedness:
            for hand_idx, hand_class in enumerate(self.results.multi_handedness):
                handedness.append(hand_class.classification[0].label) # 'Left' or 'Right'
                
        for idx, hand_landmarks in enumerate(self.results.multi_hand_landmarks):
            hand_type = handedness[idx] if idx < len(handedness) else "Unknown"
            
            # Extract specific landmark points
            landmarks_dict = {}
            for lm_idx, lm in enumerate(hand_landmarks.landmark):
                px_x = int(lm.x * w)
                px_y = int(lm.y * h)
                landmarks_dict[lm_idx] = {
                    'x': px_x,
                    'y': px_y,
                    'x_norm': lm.x,
                    'y_norm': lm.y
                }
            
            # Wrist is landmark index 0
            wrist_pos = (landmarks_dict[0]['x'], landmarks_dict[0]['y']) if 0 in landmarks_dict else (0, 0)
            wrist_norm = (landmarks_dict[0]['x_norm'], landmarks_dict[0]['y_norm']) if 0 in landmarks_dict else (0, 0)
            
            # Palm center can be approximated by averaging index, pinky base, and wrist
            palm_x = wrist_pos[0]
            palm_y = wrist_pos[1]
            palm_norm_x = wrist_norm[0]
            palm_norm_y = wrist_norm[1]
            
            palm_keys = [0, 5, 17] # wrist, index mcp, pinky mcp
            valid_keys = [k for k in palm_keys if k in landmarks_dict]
            if valid_keys:
                palm_x = int(sum(landmarks_dict[k]['x'] for k in valid_keys) / len(valid_keys))
                palm_y = int(sum(landmarks_dict[k]['y'] for k in valid_keys) / len(valid_keys))
                palm_norm_x = sum(landmarks_dict[k]['x_norm'] for k in valid_keys) / len(valid_keys)
                palm_norm_y = sum(landmarks_dict[k]['y_norm'] for k in valid_keys) / len(valid_keys)
                
            hands_data.append({
                'label': hand_type,
                'wrist': wrist_pos,
                'wrist_norm': wrist_norm,
                'palm': (palm_x, palm_y),
                'palm_norm': (palm_norm_x, palm_norm_y),
                'landmarks': landmarks_dict
            })
            
        return hands_data
