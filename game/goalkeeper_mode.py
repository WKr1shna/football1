import pygame
import cv2
import numpy as np
import time
import random
from vision.pose_detector import PoseDetector
from vision.hand_detector import HandDetector
from game.physics import BallPhysics

class GoalkeeperMode:
    """
    Mode 2: Goalkeeper Mode.
    The AI shoots penalties, and the player acts as goalkeeper.
    Uses relative wrist displacement from body center to position goalie gloves.
    """
    def __init__(self, screen, asset_manager, score_manager, cap, difficulty="MEDIUM", calibration_data=None):
        self.screen = screen
        self.assets = asset_manager
        self.scores = score_manager
        self.cap = cap
        self.difficulty = difficulty
        self.calibration_data = calibration_data or {}
        
        self.width = screen.get_width()
        self.height = screen.get_height()
        
        # Init physics & detectors
        self.physics = BallPhysics(self.width, self.height)
        self.pose_detector = PoseDetector()
        self.hand_detector = HandDetector()
        
        # State machine
        # "WAITING_SHOT" -> "BALL_FLIGHT" -> "RESULT_CELEBRATION" -> "COMPLETED"
        self.state = "WAITING_SHOT"
        self.state_timer = 0
        
        # Gameplay variables
        self.reset_attempt()
        
        # Glove size
        self.glove_radius = 45
        
        # Corner webcam view coordinates
        self.webcam_rect = pygame.Rect(self.width - 250, 20, 230, 172)
        
        # Replay recording buffer
        self.replay_data = []

    def reset_attempt(self):
        """
        Prepares for the next penalty save attempt.
        """
        self.ball_pos_3d = np.array([0.0, 0.0, 0.0])
        self.ball_screen_pos = self.physics.penalty_spot
        self.ball_radius = self.physics.start_radius
        
        self.ball_trajectory = []
        self.trajectory_index = 0
        
        # Glove positions (screen coordinates)
        # Defaults to post boundaries
        self.left_glove_pos = (self.physics.goal_left, self.physics.goal_bottom - 100)
        self.right_glove_pos = (self.physics.goal_right, self.physics.goal_bottom - 100)
        
        # AI striker shot settings
        self.shot_direction = "CENTER"
        self.shot_height = "MID"
        self.shot_power = 60
        self.shot_angle = 0.0
        
        self.outcome_text = ""
        self.outcome_color = (255, 255, 255)
        self.has_played_outcome_sound = False
        self.shot_triggered = False
        
        self.replay_data = []

    def trigger_ai_shot(self):
        """
        AI chooses shot direction, height, and power based on difficulty.
        Easy: slower shots, more predictable (mid/center).
        Hard: faster shots, aims for corners.
        """
        dirs = ["LEFT", "CENTER", "RIGHT"]
        heights = ["LOW", "MID", "HIGH"]
        
        if self.difficulty == "EASY":
            self.shot_direction = random.choice(["LEFT", "CENTER", "RIGHT"])
            self.shot_height = random.choice(["LOW", "MID"])
            self.shot_power = random.randint(40, 60)
        elif self.difficulty == "HARD":
            # Target corners more often
            self.shot_direction = random.choice(["LEFT", "RIGHT", "LEFT", "RIGHT", "CENTER"])
            self.shot_height = random.choice(["LOW", "MID", "HIGH"])
            self.shot_power = random.randint(75, 95)
        else:
            # MEDIUM
            self.shot_direction = random.choice(dirs)
            self.shot_height = random.choice(heights)
            self.shot_power = random.randint(55, 75)
            
        # Give a small curve
        self.shot_angle = random.uniform(-15.0, 15.0)
        
        self.ball_trajectory = self.physics.calculate_trajectory(
            self.shot_direction,
            self.shot_height,
            self.shot_power,
            self.shot_angle
        )
        self.trajectory_index = 0
        self.shot_triggered = True
        self.state = "BALL_FLIGHT"
        self.state_timer = time.time()
        self.assets.play_sound('kick')

    def update(self, frame_raw):
        """
        Progresses game frame and tracks goalie hands.
        """
        # Process skeletal tracking (mirror frame first)
        frame_mirrored = cv2.flip(frame_raw, 1)
        skeleton_frame = self.pose_detector.find_pose(frame_mirrored, draw=True)
        landmarks = self.pose_detector.get_landmarks(skeleton_frame.shape)
        joints = self.pose_detector.get_key_joints(landmarks)
        
        # Try to locate wrists to position gloves
        self.position_gloves(joints)
        
        # State machine
        if self.state == "WAITING_SHOT":
            # Wait 3 seconds before AI shoots
            if self.state_timer == 0:
                self.state_timer = time.time()
                
            elapsed = time.time() - self.state_timer
            if elapsed > 3.0:
                self.trigger_ai_shot()
                
        elif self.state == "BALL_FLIGHT":
            if self.trajectory_index < len(self.ball_trajectory):
                # Update ball properties
                ball_point = self.ball_trajectory[self.trajectory_index]
                self.ball_screen_pos = (ball_point['x'], ball_point['y'])
                self.ball_radius = ball_point['radius']
                self.ball_pos_3d = np.array([ball_point['x_3d'], ball_point['y_3d'], ball_point['z']])
                
                # Check for save intersection *during* the final section of flight (near goal line Z > 0.85)
                if ball_point['z'] >= 0.85:
                    self.check_for_save()
                    
                # Record frame for replay
                self.replay_data.append({
                    'ball_pos': self.ball_screen_pos,
                    'ball_radius': self.ball_radius,
                    'left_glove': self.left_glove_pos,
                    'right_glove': self.right_glove_pos,
                    'landmarks': landmarks.copy()
                })
                
                self.trajectory_index += 1
            else:
                # Trajectory finished, check final outcome
                if "SAVE" not in self.outcome_text:
                    # If not saved during flight, it entered the net
                    self.evaluate_final_outcome()
                
                self.state = "RESULT_CELEBRATION"
                self.state_timer = time.time()
                
        elif self.state == "RESULT_CELEBRATION":
            if not self.has_played_outcome_sound:
                if "SAVE" in self.outcome_text:
                    self.assets.play_sound('save')
                elif "GOAL" in self.outcome_text:
                    self.assets.play_sound('goal')
                else:
                    self.assets.play_sound('miss')
                self.has_played_outcome_sound = True
                
            if time.time() - self.state_timer > 2.5:
                # Update attempts scoreboard
                is_goal = ("GOAL" in self.outcome_text)
                is_save = ("SAVE" in self.outcome_text)
                self.scores.record_attempt(is_goal, is_save)
                
                self.state = "COMPLETED"
                
        return skeleton_frame

    def position_gloves(self, joints):
        """
        Maps the physical coordinates of the wrists to goalie gloves on the screen.
        Uses relative displacement from hips/shoulders to scale reaching range.
        """
        # Use previous position as default so gloves don't teleport when tracking is lost or during mouse fallback
        lg_x, lg_y = self.left_glove_pos
        rg_x, rg_y = self.right_glove_pos
        
        # Check if hips are visible to establish body center
        if 'left_hip' in joints and 'right_hip' in joints:
            lh = joints['left_hip']
            rh = joints['right_hip']
            
            body_center_x = (lh['x_norm'] + rh['x_norm']) / 2.0
            # Use hip height as base line
            body_y = (lh['y_norm'] + rh['y_norm']) / 2.0
            
            # Calibration parameters
            shoulder_width = self.calibration_data.get('shoulder_width', 0.15)
            arm_span = shoulder_width * 2.2 # estimated reach factor
            
            # Left wrist mapping
            if 'left_wrist' in joints:
                lw = joints['left_wrist']
                # Horizontal relative displacement (0.0 means at body center)
                # Left wrist is physically on player's left side (which is right on screen due to mirroring, 
                # but since camera is flipped, we map anatomical left_wrist directly)
                dx = lw['x_norm'] - body_center_x
                dy = body_y - lw['y_norm'] # positive means hand is raised (smaller y_norm)
                
                # Apply sensitivity gain to map physical arm movements to goal bounds
                sens_x = 1.3 / arm_span
                sens_y = 1.4 / arm_span
                
                lg_x = int(self.physics.goal_center_x + dx * sens_x * self.physics.goal_width)
                lg_y = int(self.physics.goal_bottom - (dy + 0.1) * sens_y * self.physics.goal_height)
                
            # Right wrist mapping
            if 'right_wrist' in joints:
                rw = joints['right_wrist']
                dx = rw['x_norm'] - body_center_x
                dy = body_y - rw['y_norm']
                
                sens_x = 1.3 / arm_span
                sens_y = 1.4 / arm_span
                
                rg_x = int(self.physics.goal_center_x + dx * sens_x * self.physics.goal_width)
                rg_y = int(self.physics.goal_bottom - (dy + 0.1) * sens_y * self.physics.goal_height)
        
        # Clamp glove coordinates within reasonable bounds around the goal post area
        margin_x = 80
        margin_y = 60
        self.left_glove_pos = (
            np.clip(lg_x, self.physics.goal_left - margin_x, self.physics.goal_right + margin_x),
            np.clip(lg_y, self.physics.goal_top - margin_y, self.physics.goal_bottom + margin_y)
        )
        self.right_glove_pos = (
            np.clip(rg_x, self.physics.goal_left - margin_x, self.physics.goal_right + margin_x),
            np.clip(rg_y, self.physics.goal_top - margin_y, self.physics.goal_bottom + margin_y)
        )

    def check_for_save(self):
        """
        Computes distance between the ball and both goalie gloves.
        Triggers a save if they overlap.
        """
        bx, by = self.ball_screen_pos
        
        # Calculate Euclidean distances
        dist_l = np.sqrt((bx - self.left_glove_pos[0])**2 + (by - self.left_glove_pos[1])**2)
        dist_r = np.sqrt((bx - self.right_glove_pos[0])**2 + (by - self.right_glove_pos[1])**2)
        
        # Check collision threshold (ball radius + glove radius)
        threshold = self.ball_radius + self.glove_radius
        
        if dist_l < threshold or dist_r < threshold:
            self.outcome_text = "GREAT SAVE!!!"
            self.outcome_color = (100, 255, 100)
            self.trajectory_index = len(self.ball_trajectory) # stop ball animation immediately

    def evaluate_final_outcome(self):
        """
        Fell through without save during flight: check if goal or miss.
        """
        x_3d = self.ball_pos_3d[0]
        y_3d = self.ball_pos_3d[1]
        
        inside = self.physics.is_inside_goal(x_3d, y_3d)
        
        if inside:
            self.outcome_text = "GOAL CONCEDED!"
            self.outcome_color = (255, 100, 100)
        else:
            self.outcome_text = "SHOT WIDE / MISSED!"
            self.outcome_color = (200, 200, 200)

    def draw(self, skeleton_frame):
        """
        Draws the game assets on screen.
        """
        # 1. Background Stadium
        self.screen.blit(self.assets.images['stadium'], (0, 0))
        
        # 2. Draw Net / Goal Image
        if self.assets.images['goal']:
            self.screen.blit(self.assets.images['goal'], (self.physics.goal_left, self.physics.goal_top))
        else:
            self._draw_procedural_goal()
            
        # 3. Draw semi-transparent player skeleton silhouette in goal area (Kinect style)
        self._draw_goalie_body_overlay(skeleton_frame)
        
        # 4. Draw Goalkeeper Gloves (Large circles or assets)
        self._draw_gloves()
        
        # 5. Draw Ball
        if self.state in ["BALL_FLIGHT", "RESULT_CELEBRATION"]:
            ball_img = self.assets.images['ball']
            scaled_ball = pygame.transform.scale(ball_img, (self.ball_radius * 2, self.ball_radius * 2))
            ball_rect = scaled_ball.get_rect(center=self.ball_screen_pos)
            self.screen.blit(scaled_ball, ball_rect.topleft)
            
        # 6. Draw HUD dashboard
        self._draw_hud()
        
        # 7. Draw prompts & outcome celebrations
        if self.state == "WAITING_SHOT":
            # Countdown or shoot notice
            elapsed = time.time() - self.state_timer
            cd = int(4.0 - elapsed)
            cd = max(1, cd)
            prompt_text = f"AI PREPARING SHOT... GET READY in {cd}..."
            prompt_surf = self.assets.fonts['header'].render(prompt_text, True, (255, 230, 80))
            self.screen.blit(prompt_surf, (self.width // 2 - prompt_surf.get_width() // 2, 80))
            
        elif self.state == "RESULT_CELEBRATION":
            out_surf = self.assets.fonts['title'].render(self.outcome_text, True, self.outcome_color)
            self.screen.blit(out_surf, (self.width // 2 - out_surf.get_width() // 2, 130))
            
            # Show details of shot
            det_text = f"AI Shot: {self.shot_direction} | Height: {self.shot_height} | Power: {self.shot_power}%"
            det_surf = self.assets.fonts['body'].render(det_text, True, (255, 255, 255))
            self.screen.blit(det_surf, (self.width // 2 - det_surf.get_width() // 2, 200))

        # 8. Draw Scaled Webcam overlay in top-right corner
        pygame.draw.rect(self.screen, (50, 100, 255), self.webcam_rect, 2, border_radius=10)
        frame_rgb = cv2.cvtColor(skeleton_frame, cv2.COLOR_BGR2RGB)
        frame_surf = pygame.image.frombuffer(frame_rgb.tobytes(), frame_rgb.shape[1::-1], "RGB")
        frame_scaled = pygame.transform.scale(frame_surf, (self.webcam_rect.width - 4, self.webcam_rect.height - 4))
        self.screen.blit(frame_scaled, (self.webcam_rect.x + 2, self.webcam_rect.y + 2))

    def _draw_procedural_goal(self):
        """
        Draws posts and net mesh.
        """
        gl = self.physics.goal_left
        gr = self.physics.goal_right
        gt = self.physics.goal_top
        gb = self.physics.goal_bottom
        
        pygame.draw.line(self.screen, (240, 240, 240), (gl, gb), (gl, gt), 8)
        pygame.draw.line(self.screen, (240, 240, 240), (gr, gb), (gr, gt), 8)
        pygame.draw.line(self.screen, (240, 240, 240), (gl - 4, gt), (gr + 4, gt), 8)
        
        # Net netting
        net_surf = pygame.Surface((self.physics.goal_width, self.physics.goal_height), pygame.SRCALPHA)
        net_surf.fill((255, 255, 255, 5))
        grid_space = 25
        for x in range(0, self.physics.goal_width, grid_space):
            pygame.draw.line(net_surf, (240, 240, 240, 20), (x, 0), (x, self.physics.goal_height), 1)
        for y in range(0, self.physics.goal_height, grid_space):
            pygame.draw.line(net_surf, (240, 240, 240, 20), (0, y), (self.physics.goal_width, y), 1)
        self.screen.blit(net_surf, (gl, gt))

    def _draw_gloves(self):
        """
        Draws the virtual goalie gloves.
        """
        # Outer neon glow
        pygame.draw.circle(self.screen, (255, 100, 100, 100), self.left_glove_pos, self.glove_radius + 4, 3)
        pygame.draw.circle(self.screen, (100, 150, 255, 100), self.right_glove_pos, self.glove_radius + 4, 3)
        
        # Inner glove circle
        pygame.draw.circle(self.screen, (220, 50, 50), self.left_glove_pos, self.glove_radius)
        pygame.draw.circle(self.screen, (50, 100, 220), self.right_glove_pos, self.glove_radius)
        
        # Core highlight
        pygame.draw.circle(self.screen, (255, 255, 255), self.left_glove_pos, self.glove_radius - 12)
        pygame.draw.circle(self.screen, (255, 255, 255), self.right_glove_pos, self.glove_radius - 12)
        
        # Write "L" and "R"
        l_text = self.assets.fonts['body'].render("L", True, (220, 50, 50))
        r_text = self.assets.fonts['body'].render("R", True, (50, 100, 220))
        
        self.screen.blit(l_text, (self.left_glove_pos[0] - l_text.get_width()//2, self.left_glove_pos[1] - l_text.get_height()//2))
        self.screen.blit(r_text, (self.right_glove_pos[0] - r_text.get_width()//2, self.right_glove_pos[1] - r_text.get_height()//2))

    def _draw_goalie_body_overlay(self, skeleton_frame):
        """
        Draws a skeletal outline of the player centered in the goal.
        Helps the player visualize their posture and reach.
        """
        # Map all pose detector landmarks to the screen goal area
        if not self.pose_detector.results or not self.pose_detector.results.pose_landmarks:
            return
            
        pose_lms = self.pose_detector.results.pose_landmarks.landmark
        
        # We draw skeletal connections scaled to the goal width & height
        connections = self.pose_detector.mp_pose.POSE_CONNECTIONS
        
        # Body calibration metrics
        shoulder_width = self.calibration_data.get('shoulder_width', 0.15)
        arm_span = shoulder_width * 2.2
        
        h, w = self.height, self.width
        
        # Draw connections
        for connection in connections:
            start_idx, end_idx = connection
            lm_start = pose_lms[start_idx]
            lm_end = pose_lms[end_idx]
            
            # Require minimum visibility
            if lm_start.visibility < 0.4 or lm_end.visibility < 0.4:
                continue
                
            # Find hip center as reference center
            lh = pose_lms[self.pose_detector.mp_pose.PoseLandmark.LEFT_HIP.value]
            rh = pose_lms[self.pose_detector.mp_pose.PoseLandmark.RIGHT_HIP.value]
            body_center_x = (lh.x + rh.x) / 2.0
            body_y = (lh.y + rh.y) / 2.0
            
            # Map start point
            dx_s = lm_start.x - body_center_x
            dy_s = body_y - lm_start.y
            
            sens_x = 1.3 / arm_span
            sens_y = 1.4 / arm_span
            
            sx = int(self.physics.goal_center_x + dx_s * sens_x * self.physics.goal_width)
            sy = int(self.physics.goal_bottom - (dy_s + 0.1) * sens_y * self.physics.goal_height)
            
            # Map end point
            dx_e = lm_end.x - body_center_x
            dy_e = body_y - lm_end.y
            
            ex = int(self.physics.goal_center_x + dx_e * sens_x * self.physics.goal_width)
            ey = int(self.physics.goal_bottom - (dy_e + 0.1) * sens_y * self.physics.goal_height)
            
            # Draw overlay skeleton line
            pygame.draw.line(self.screen, (0, 255, 200, 150), (sx, sy), (ex, ey), 3)

    def _draw_hud(self):
        """
        Renders HUD dashboard.
        """
        # Top Left Score Panel
        hud_panel = pygame.Surface((320, 110), pygame.SRCALPHA)
        hud_panel.fill((15, 15, 25, 200))
        pygame.draw.rect(hud_panel, (255, 100, 100, 128), (0, 0, 320, 110), 2, border_radius=10)
        self.screen.blit(hud_panel, (20, 20))
        
        # Text entries
        txt_title = self.assets.fonts['hud'].render("GOALKEEPER CHALLENGE", True, (255, 100, 100))
        txt_score = self.assets.fonts['hud'].render(f"SAVES: {self.scores.saves} / {self.scores.attempts}", True, (255, 255, 255))
        txt_remaining = self.assets.fonts['hud'].render(f"SHOTS REMAINING: {self.scores.max_attempts - self.scores.attempts}", True, (200, 200, 200))
        
        self.screen.blit(txt_title, (35, 30))
        self.screen.blit(txt_score, (35, 55))
        self.screen.blit(txt_remaining, (35, 80))
        
        # Draw difficulty indicator in bottom-left corner
        diff_panel = pygame.Surface((180, 45), pygame.SRCALPHA)
        diff_panel.fill((15, 15, 25, 180))
        pygame.draw.rect(diff_panel, (100, 100, 100, 100), (0, 0, 180, 45), 1, border_radius=5)
        self.screen.blit(diff_panel, (20, self.height - 65))
        
        diff_lbl = self.assets.fonts['hud'].render(f"DIFF: {self.difficulty}", True, (255, 230, 80))
        self.screen.blit(diff_lbl, (35, self.height - 53))
