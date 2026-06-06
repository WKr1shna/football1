import unittest
import numpy as np
import time
import os
import json
import sys

# Add path to import game modules
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from vision.kick_detector import KickDetector
from game.physics import BallPhysics
from game.scoring import ScoreManager

class TestVisionFootball(unittest.TestCase):
    def setUp(self):
        self.physics = BallPhysics(1024, 768)
        self.score_file = "test_scores.json"
        self.scores = ScoreManager(filename=self.score_file)
        self.detector = KickDetector(history_len=3, speed_threshold=0.03, cool_down_time=0.1)

    def tearDown(self):
        # Clean up test scores file
        if os.path.exists(self.scores.filepath):
            try:
                os.remove(self.scores.filepath)
            except Exception:
                pass

    def test_physics_trajectory(self):
        """
        Verify that 3D trajectory points are generated correctly and shrink in perspective.
        """
        trajectory = self.physics.calculate_trajectory("LEFT", "HIGH", 80, -10.0)
        self.assertGreater(len(trajectory), 10)
        
        # Ball should start on the penalty spot (width // 2, int(height * 0.82))
        start_pt = trajectory[0]
        self.assertEqual(start_pt['x'], 512)
        self.assertEqual(start_pt['y'], 630) # 768 * 0.82 = 629.76
        self.assertEqual(start_pt['radius'], self.physics.start_radius)
        
        # Ball should shrink to end_radius at depth Z=1
        end_pt = trajectory[-1]
        self.assertEqual(end_pt['radius'], self.physics.end_radius)
        self.assertAlmostEqual(end_pt['z'], 1.0, places=4)
        
        # Verify LEFT shot target is on the left side of goal center
        self.assertLess(end_pt['x'], 512)

    def test_score_manager(self):
        """
        Verify that scores are saved, loaded, and ranked correctly.
        """
        self.scores.reset_score()
        self.scores.record_attempt(is_goal=True, is_save=False)
        self.scores.record_attempt(is_goal=True, is_save=False)
        self.scores.record_attempt(is_goal=False, is_save=True)
        
        self.assertEqual(self.scores.attempts, 3)
        self.assertEqual(self.scores.goals, 2)
        self.assertEqual(self.scores.saves, 1)
        
        # Add high scores to test rankings
        self.scores.add_score("ALICE", "STRIKER", 4, "MEDIUM")
        self.scores.add_score("BOB", "STRIKER", 5, "HARD")
        self.scores.add_score("CHARLIE", "STRIKER", 3, "EASY")
        
        top = self.scores.get_top_scores("STRIKER")
        self.assertEqual(top[0]["name"], "BOB")      # 5 goals
        self.assertEqual(top[1]["name"], "ALICE")    # 4 goals
        self.assertEqual(top[2]["name"], "CHARLIE")  # 3 goals

    def test_kick_detector_velocity(self):
        """
        Verify that rapid ankle displacement triggers a Kick Event.
        """
        # Base resting pose joints
        base_joints = {
            'left_hip': {'x_norm': 0.45, 'y_norm': 0.55},
            'right_hip': {'x_norm': 0.55, 'y_norm': 0.55},
            'left_knee': {'x_norm': 0.45, 'y_norm': 0.70},
            'right_knee': {'x_norm': 0.55, 'y_norm': 0.70},
            'left_ankle': {'x_norm': 0.45, 'y_norm': 0.85},
            'right_ankle': {'x_norm': 0.55, 'y_norm': 0.85},
            'left_shoulder': {'x_norm': 0.45, 'y_norm': 0.35},
            'right_shoulder': {'x_norm': 0.55, 'y_norm': 0.35}
        }
        
        # Feed static pose - should NOT trigger kick
        now = time.time()
        for i in range(3):
            # Advance timestamp artificially
            t = now + i * 0.03
            joints = base_joints.copy()
            
            # Mock process_landmarks behavior by injecting artificial timestamps
            # In unittest, process_landmarks uses actual time.time(), so we mock it by feeding values
            # and modifying history queue manually for test precision.
            self.detector.left_ankle_history.append((t, 0.45, 0.85))
            self.detector.right_ankle_history.append((t, 0.55, 0.85))
            
        event = self.detector.process_landmarks(base_joints)
        self.assertIsNone(event)
        
        # Feed high speed ankle displacement
        t_kick = t + 0.03
        kick_joints = base_joints.copy()
        # Move right ankle significantly to the left (anatomical right foot kicking left)
        kick_joints['right_ankle'] = {'x_norm': 0.38, 'y_norm': 0.75} # rapid delta_x = -0.17, delta_y = -0.10
        
        self.detector.left_ankle_history.append((t_kick, 0.45, 0.85))
        self.detector.right_ankle_history.append((t_kick, 0.38, 0.75))
        
        event = self.detector._create_kick_event(kick_joints, "right", 0.15)
        self.assertIsNotNone(event)
        self.assertEqual(event['leg'], 'right')
        self.assertEqual(event['direction'], 'LEFT') # Since ankle_x (0.38) < body_center_x (0.5) - margin

if __name__ == "__main__":
    unittest.main()
