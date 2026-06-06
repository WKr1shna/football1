import pygame
import cv2
import numpy as np
import time
from vision.pose_detector import PoseDetector
from vision.kick_detector import KickDetector
from game.physics import BallPhysics

class PenaltyMode:
    """
    Mode 1: Penalty Kicker.
    Player acts as striker, kicks physically to shoot.
    AI goalkeeper dives to defend based on difficulty.
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
        
        # Pull calibration values if available
        sh_width = self.calibration_data.get('shoulder_width', 0.15)
        leg_len = self.calibration_data.get('leg_length', 0.40)
        self.kick_detector = KickDetector()
        self.kick_detector.update_calibration(sh_width, leg_len)
        
        # Game state machine for Penalty Mode
        # "WAITING_KICK" -> "BALL_FLIGHT" -> "RESULT_CELEBRATION" -> "COMPLETED"
        self.state = "WAITING_KICK"
        self.state_timer = 0
        
        # Gameplay variables
        self.reset_attempt()
        
        # Corner webcam view coordinates
        self.webcam_rect = pygame.Rect(self.width - 250, 20, 230, 172)
        
        # Recording buffer for replay system
        self.replay_data = []

    def reset_attempt(self):
        """
        Prepares for the next penalty kick.
        """
        self.ball_pos_3d = np.array([0.0, 0.0, 0.0])
        self.ball_screen_pos = self.physics.penalty_spot
        self.ball_radius = self.physics.start_radius
        
        self.ball_trajectory = []
        self.trajectory_index = 0
        
        # Goalkeeper animation variables
        # Base positioning (center of goal line)
        self.keeper_x = self.width // 2
        self.keeper_y = self.physics.goal_bottom - 20
        self.keeper_target_x = self.width // 2
        self.keeper_target_y = self.physics.goal_bottom - 20
        self.keeper_dive_dir = "CENTER"
        self.keeper_rotation = 0.0
        
        # AI decision parameters
        self.ai_delay_frames = 0
        self.ai_decision_made = False
        self.ai_dive_trigger_index = 0
        
        # Kicked parameters
        self.kick_info = None
        self.outcome_text = ""
        self.outcome_color = (255, 255, 255)
        self.has_played_outcome_sound = False
        
        # Clear replay buffer
        self.replay_data = []

    def determine_ai_goalkeeper_dive(self, kick_direction, kick_height):
        """
        Calculates keeper dive path based on chosen difficulty.
        Easy: 20% save chance.
        Medium: 50% save chance.
        Hard: 80% save chance.
        Also sets reaction delay (fewer frames = quicker reaction).
        """
        save_odds = {"EASY": 0.20, "MEDIUM": 0.50, "HARD": 0.80}
        odds = save_odds.get(self.difficulty, 0.50)
        
        # Goalkeeper reaction delay (in frames of ball flight)
        # Easy: goalkeeper starts diving late (e.g. 50% into ball flight)
        # Hard: goalkeeper starts diving immediately (e.g. 15% into ball flight)
        delay_ratios = {"EASY": 0.55, "MEDIUM": 0.35, "HARD": 0.15}
        delay_ratio = delay_ratios.get(self.difficulty, 0.35)
        
        total_flight_frames = len(self.ball_trajectory)
        self.ai_dive_trigger_index = int(total_flight_frames * delay_ratio)
        
        # Determine if AI guesses correctly
        guess_correctly = (np.random.random() < odds)
        
        if guess_correctly:
            self.keeper_dive_dir = kick_direction
        else:
            # Guess wrong: choose a random incorrect direction
            dirs = ["LEFT", "CENTER", "RIGHT"]
            dirs.remove(kick_direction)
            self.keeper_dive_dir = np.random.choice(dirs)

        # Map keeper's final destination in pixels
        goal_w = self.physics.goal_width
        goal_h = self.physics.goal_height
        
        if self.keeper_dive_dir == "LEFT":
            self.keeper_target_x = self.physics.goal_left + int(goal_w * 0.15)
            # Match high/low
            self.keeper_target_y = self.physics.goal_bottom - int(goal_h * (0.6 if kick_height == "HIGH" else 0.2))
        elif self.keeper_dive_dir == "RIGHT":
            self.keeper_target_x = self.physics.goal_right - int(goal_w * 0.15)
            self.keeper_target_y = self.physics.goal_bottom - int(goal_h * (0.6 if kick_height == "HIGH" else 0.2))
        else:
            # Center
            self.keeper_target_x = self.physics.goal_center_x
            self.keeper_target_y = self.physics.goal_bottom - int(goal_h * (0.4 if kick_height == "HIGH" else 0.1))

    def update(self, frame_raw):
        """
        Progresses game frame and tracks players.
        """
        # Process skeletal tracking
        frame_mirrored = cv2.flip(frame_raw, 1)
        skeleton_frame = self.pose_detector.find_pose(frame_mirrored, draw=True)
        landmarks = self.pose_detector.get_landmarks(skeleton_frame.shape)
        joints = self.pose_detector.get_key_joints(landmarks)
        
        # Handle state transitions
        if self.state == "WAITING_KICK":
            # Scan for kicking speed triggers
            kick_event = self.kick_detector.process_landmarks(joints)
            if kick_event:
                self.assets.play_sound('kick')
                self.kick_info = kick_event
                
                # Setup ball physics trajectory
                self.ball_trajectory = self.physics.calculate_trajectory(
                    kick_event['direction'],
                    kick_event['height'],
                    kick_event['power'],
                    kick_event['kick_angle']
                )
                self.trajectory_index = 0
                
                # Goalkeeper AI calculations
                self.determine_ai_goalkeeper_dive(kick_event['direction'], kick_event['height'])
                
                self.state = "BALL_FLIGHT"
                
        elif self.state == "BALL_FLIGHT":
            # Progress ball along Bezier path
            if self.trajectory_index < len(self.ball_trajectory):
                # Record details for replay
                ball_point = self.ball_trajectory[self.trajectory_index]
                self.ball_screen_pos = (ball_point['x'], ball_point['y'])
                self.ball_radius = ball_point['radius']
                self.ball_pos_3d = np.array([ball_point['x_3d'], ball_point['y_3d'], ball_point['z']])
                
                # Goalkeeper dive movement animation
                if self.trajectory_index >= self.ai_dive_trigger_index:
                    # Animate goalkeeper towards target
                    # Move keeper incrementally
                    anim_progress = (self.trajectory_index - self.ai_dive_trigger_index) / (len(self.ball_trajectory) - self.ai_dive_trigger_index)
                    
                    start_x = self.width // 2
                    start_y = self.physics.goal_bottom - 20
                    self.keeper_x = int(start_x + (self.keeper_target_x - start_x) * anim_progress)
                    self.keeper_y = int(start_y + (self.keeper_target_y - start_y) * anim_progress)
                    
                    # Add rotation slide
                    if self.keeper_dive_dir == "LEFT":
                        self.keeper_rotation = 45.0 * anim_progress
                    elif self.keeper_dive_dir == "RIGHT":
                        self.keeper_rotation = -45.0 * anim_progress
                    else:
                        self.keeper_rotation = 0.0
                
                # Record frame data for replay
                self.replay_data.append({
                    'ball_pos': self.ball_screen_pos,
                    'ball_radius': self.ball_radius,
                    'keeper_pos': (self.keeper_x, self.keeper_y),
                    'keeper_rotation': self.keeper_rotation,
                    'landmarks': landmarks.copy()
                })
                
                self.trajectory_index += 1
            else:
                # Ball flight finished, evaluate outcome
                self.evaluate_outcome()
                self.state = "RESULT_CELEBRATION"
                self.state_timer = time.time()
                
        elif self.state == "RESULT_CELEBRATION":
            # Play appropriate sound once
            if not self.has_played_outcome_sound:
                if "GOAL" in self.outcome_text:
                    self.assets.play_sound('goal')
                    self.assets.play_sound('cheer')
                elif "SAVE" in self.outcome_text:
                    self.assets.play_sound('save')
                else:
                    self.assets.play_sound('miss')
                self.has_played_outcome_sound = True
                
            # Wait 2.5 seconds before moving to REPLAY or NEXT shot
            if time.time() - self.state_timer > 2.5:
                # Update attempts scoreboard
                is_goal = ("GOAL" in self.outcome_text)
                is_save = ("SAVE" in self.outcome_text)
                self.scores.record_attempt(is_goal, is_save)
                
                # Move state machine to final completed
                self.state = "COMPLETED"
                
        return skeleton_frame

    def evaluate_outcome(self):
        """
        Checks if ball enters goal or gets blocked by goalkeeper.
        """
        x_3d = self.ball_pos_3d[0]
        y_3d = self.ball_pos_3d[1]
        
        # Check if inside post boundaries
        inside = self.physics.is_inside_goal(x_3d, y_3d)
        
        if not inside:
            self.outcome_text = "MISS! WIDE / HIGH"
            self.outcome_color = (255, 100, 100)
            return

        # Goal is inside, did goalie save it?
        # Check 2D distance between ball center and keeper sprite collision box
        # Goalkeeper covers a bounding circle around (keeper_x, keeper_y)
        ball_x, ball_y = self.ball_screen_pos
        dx = ball_x - self.keeper_x
        dy = ball_y - self.keeper_y
        dist = np.sqrt(dx**2 + dy**2)
        
        # Goalkeeper save radius
        # Harder modes have larger keeper coverage area
        save_radii = {"EASY": 60, "MEDIUM": 75, "HARD": 90}
        save_r = save_radii.get(self.difficulty, 75)
        
        if dist < save_r and self.keeper_dive_dir == self.kick_info['direction']:
            # Goalkeeper guessed right and reached it
            self.outcome_text = "SAVED BY KEEPER!"
            self.outcome_color = (100, 150, 255)
        else:
            self.outcome_text = "GOAL!!!"
            self.outcome_color = (100, 255, 100)

    def draw(self, skeleton_frame):
        """
        Renders gameplay graphics.
        """
        # 1. Background Stadium
        self.screen.blit(self.assets.images['stadium'], (0, 0))
        
        # 2. Draw Net / Goal Image
        if self.assets.images['goal']:
            self.screen.blit(self.assets.images['goal'], (self.physics.goal_left, self.physics.goal_top))
        else:
            self._draw_procedural_goal()
            
        # 3. Draw Goalkeeper
        self._draw_goalkeeper()
        
        # 4. Draw Ball (if not kicked, draw at penalty spot. Otherwise, animate)
        if self.state in ["WAITING_KICK", "BALL_FLIGHT", "RESULT_CELEBRATION"]:
            ball_img = self.assets.images['ball']
            scaled_ball = pygame.transform.scale(ball_img, (self.ball_radius * 2, self.ball_radius * 2))
            ball_rect = scaled_ball.get_rect(center=self.ball_screen_pos)
            self.screen.blit(scaled_ball, ball_rect.topleft)

        # 5. Draw HUD overlays
        self._draw_hud()
        
        # 6. Draw AI shot prediction overlay (if WAITING_KICK or early BALL_FLIGHT)
        if self.state == "WAITING_KICK":
            # Prompt instructions
            prompt_surf = self.assets.fonts['header'].render("PREPARE TO KICK! SWING LEG SPEEDILY", True, (255, 255, 255))
            self.screen.blit(prompt_surf, (self.width // 2 - prompt_surf.get_width() // 2, 80))
        elif self.state == "BALL_FLIGHT" and self.kick_info:
            pred_text = f"AI SHOT PREDICTION: {self.kick_info['prediction']} ({int(self.kick_info['confidence']*100)}% Conf)"
            pred_surf = self.assets.fonts['body'].render(pred_text, True, (255, 230, 80))
            self.screen.blit(pred_surf, (self.width // 2 - pred_surf.get_width() // 2, 85))
            
        # 7. Draw Outcome Text (during celebration)
        if self.state == "RESULT_CELEBRATION":
            out_surf = self.assets.fonts['title'].render(self.outcome_text, True, self.outcome_color)
            self.screen.blit(out_surf, (self.width // 2 - out_surf.get_width() // 2, 130))
            
            # Show details of kick
            det_text = f"Power: {self.kick_info['power']}% | Dir: {self.kick_info['direction']} | Height: {self.kick_info['height']}"
            det_surf = self.assets.fonts['body'].render(det_text, True, (255, 255, 255))
            self.screen.blit(det_surf, (self.width // 2 - det_surf.get_width() // 2, 200))

        # 8. Draw Scaled Webcam overlay in the top-right corner
        pygame.draw.rect(self.screen, (50, 100, 255), self.webcam_rect, 2, border_radius=10)
        frame_rgb = cv2.cvtColor(skeleton_frame, cv2.COLOR_BGR2RGB)
        frame_surf = pygame.image.frombuffer(frame_rgb.tobytes(), frame_rgb.shape[1::-1], "RGB")
        frame_scaled = pygame.transform.scale(frame_surf, (self.webcam_rect.width - 4, self.webcam_rect.height - 4))
        self.screen.blit(frame_scaled, (self.webcam_rect.x + 2, self.webcam_rect.y + 2))

    def _draw_procedural_goal(self):
        """
        Draws realistic white post rails and crossbar.
        """
        gl = self.physics.goal_left
        gr = self.physics.goal_right
        gt = self.physics.goal_top
        gb = self.physics.goal_bottom
        
        # Posts & Bar (White)
        pygame.draw.line(self.screen, (240, 240, 240), (gl, gb), (gl, gt), 8) # Left Post
        pygame.draw.line(self.screen, (240, 240, 240), (gr, gb), (gr, gt), 8) # Right Post
        pygame.draw.line(self.screen, (240, 240, 240), (gl - 4, gt), (gr + 4, gt), 8) # Crossbar
        
        # Net netting (subtle grid lines inside)
        net_surf = pygame.Surface((self.physics.goal_width, self.physics.goal_height), pygame.SRCALPHA)
        net_surf.fill((255, 255, 255, 5)) # very light white overlay
        
        grid_space = 25
        for x in range(0, self.physics.goal_width, grid_space):
            pygame.draw.line(net_surf, (240, 240, 240, 25), (x, 0), (x, self.physics.goal_height), 1)
        for y in range(0, self.physics.goal_height, grid_space):
            pygame.draw.line(net_surf, (240, 240, 240, 25), (0, y), (self.physics.goal_width, y), 1)
            
        self.screen.blit(net_surf, (gl, gt))

    def _draw_goalkeeper(self):
        """
        Draws the goalkeeper sprite, rotated and positioned dynamically.
        """
        keeper_img = self.assets.images['goalkeeper']
        
        # Rotate keeper if diving
        if self.keeper_rotation != 0.0:
            rotated_keeper = pygame.transform.rotate(keeper_img, self.keeper_rotation)
        else:
            rotated_keeper = keeper_img
            
        keeper_rect = rotated_keeper.get_rect(center=(self.keeper_x, self.keeper_y))
        self.screen.blit(rotated_keeper, keeper_rect.topleft)

    def _draw_hud(self):
        """
        HUD dashboard overlay showing goals, saves, FPS, attempts left.
        """
        # Top Left Score Panel
        hud_panel = pygame.Surface((320, 110), pygame.SRCALPHA)
        hud_panel.fill((15, 15, 25, 200))
        pygame.draw.rect(hud_panel, (50, 120, 255, 128), (0, 0, 320, 110), 2, border_radius=10)
        self.screen.blit(hud_panel, (20, 20))
        
        # Text entries
        txt_title = self.assets.fonts['hud'].render("STRIKER CHALLENGE", True, (50, 200, 255))
        txt_score = self.assets.fonts['hud'].render(f"GOALS: {self.scores.goals} / {self.scores.attempts}", True, (255, 255, 255))
        txt_remaining = self.assets.fonts['hud'].render(f"SHOTS REMAINING: {self.scores.max_attempts - self.scores.attempts}", True, (200, 200, 200))
        
        self.screen.blit(txt_title, (35, 30))
        self.screen.blit(txt_score, (35, 55))
        self.screen.blit(txt_remaining, (35, 80))
        
        # Draw difficulty indicator in bottom-left corner
        diff_panel = pygame.Surface((180, 45), pygame.SRCALPHA)
        diff_panel.fill((15, 15, 25, 180))
        pygame.draw.rect(diff_panel, (100, 100, 100, 100), (0, 0, 180, 45), 1, border_radius=5)
        self.screen.blit(diff_panel, (20, self.height - 65))
        
        diff_lbl = self.assets.fonts['hud'].render(f"AI: {self.difficulty}", True, (100, 255, 150))
        self.screen.blit(diff_lbl, (35, self.height - 53))
