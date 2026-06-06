import pygame
import cv2
import sys
import os
import time
import numpy as np
import random

# Add parent directory to path to ensure relative imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from game.assets import AssetManager
from game.scoring import ScoreManager
from game.menu import MainMenu
from game.penalty_mode import PenaltyMode
from game.goalkeeper_mode import GoalkeeperMode
from vision.pose_detector import PoseDetector
from game.physics import BallPhysics

class GameOrchestrator:
    """
    Main orchestrator for Vision Football Penalty Challenge.
    Manages the Pygame loop, camera acquisition, and state transitions:
    MENU, CALIBRATION, PLAY_STRIKER, PLAY_GOALKEEPER, REPLAY, LEADERBOARD, NAME_INPUT.
    """
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Vision Football Penalty Challenge")
        
        # Resolution setting
        self.screen_width = 1024
        self.screen_height = 768
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # Load resources
        self.assets = AssetManager(self.screen_width, self.screen_height)
        self.scores = ScoreManager()
        self.physics = BallPhysics(self.screen_width, self.screen_height)
        
        # Initialize video capture
        self.cap = None
        self.webcam_connected = False
        self.init_webcam()
        
        # Shared Calibration Data
        self.calibration_data = {
            'shoulder_width': 0.15,
            'leg_length': 0.40,
            'hip_height': 0.60
        }
        self.calibrated = False
        
        # Game State
        # States: "MENU", "CALIBRATION", "PLAY_STRIKER", "PLAY_GOALKEEPER", "REPLAY", "LEADERBOARD", "NAME_INPUT"
        self.state = "MENU"
        self.menu = MainMenu(self.screen, self.assets, self.scores, self.cap)
        
        # Active gameplay instances
        self.active_striker_game = None
        self.active_keeper_game = None
        
        # Replay system buffers
        self.replay_buffer = []
        self.replay_mode_type = "STRIKER" # or "GOALKEEPER"
        self.replay_frame_index = 0
        
        # Keyboard fallback indicators
        self.fallback_mouse_control = not self.webcam_connected
        
        # Leaderboard name input vars
        self.player_name = ""
        self.leaderboard_mode = "STRIKER"
        self.qualifying_score = 0
        
        # FPS Tracker
        self.fps_font = pygame.font.SysFont("monospace", 15)

    def init_webcam(self):
        """
        Attempts to bind the system camera.
        Sets up fallback flag if webcam is missing.
        """
        print("Initializing webcam...")
        # Index 0 is default camera
        self.cap = cv2.VideoCapture(0)
        
        # Set resolution for lower latency
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if self.cap.isOpened():
            self.webcam_connected = True
            print("Webcam successfully initialized.")
        else:
            self.webcam_connected = False
            self.fallback_mouse_control = True
            print("WARNING: Webcam not detected. Enabling mouse/keyboard fallback mode.")

    def run(self):
        """
        Core game execution loop.
        """
        running = True
        while running:
            # 1. Fetch camera frame
            frame = None
            if self.webcam_connected:
                ret, raw_frame = self.cap.read()
                if ret:
                    frame = raw_frame
                else:
                    # Camera connection lost
                    self.webcam_connected = False
                    self.fallback_mouse_control = True
            
            # If camera not connected, construct a black dummy frame
            if frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Webcam Disconnected - Mouse Fallback Enabled", (30, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # 2. Process inputs / Pygame Events
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    running = False
                    
                # Handle global keys
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.state != "MENU":
                            self.state = "MENU"
                            self.assets.play_sound('kick')
                            
                # Route events based on state
                if self.state == "MENU":
                    action = self.menu.handle_event(event)
                    if action:
                        self.handle_menu_action(action)
                        
                elif self.state == "LEADERBOARD":
                    if event.type == pygame.KEYDOWN:
                        self.state = "MENU"
                        self.assets.play_sound('kick')
                        
                elif self.state == "NAME_INPUT":
                    self.handle_name_input_events(event)

            # 3. Update & render screen based on State Machine
            if self.state == "MENU":
                self.menu.draw(frame)
                
            elif self.state == "CALIBRATION":
                self.run_calibration_loop(frame)
                
            elif self.state == "PLAY_STRIKER":
                self.run_striker_loop(frame)
                
            elif self.state == "PLAY_GOALKEEPER":
                self.run_keeper_loop(frame)
                
            elif self.state == "REPLAY":
                self.run_replay_loop()
                
            elif self.state == "LEADERBOARD":
                self.draw_leaderboard_screen()
                
            elif self.state == "NAME_INPUT":
                self.draw_name_input_screen()
                
            # Draw FPS Overlay
            self.draw_fps()
            
            # Flip display buffer
            pygame.display.flip()
            self.clock.tick(60)
            
        # Clean up
        if self.cap and self.cap.isOpened():
            self.cap.release()
        pygame.quit()
        sys.exit()

    def handle_menu_action(self, action):
        """
        Transitions to different screens from Main Menu.
        """
        if action == "STRIKER":
            self.scores.reset_score()
            self.state = "PLAY_STRIKER"
            self.active_striker_game = PenaltyMode(
                self.screen, self.assets, self.scores, self.cap,
                difficulty=self.menu.get_difficulty(),
                calibration_data=self.calibration_data
            )
        elif action == "GOALKEEPER":
            self.scores.reset_score()
            self.state = "PLAY_GOALKEEPER"
            self.active_keeper_game = GoalkeeperMode(
                self.screen, self.assets, self.scores, self.cap,
                difficulty=self.menu.get_difficulty(),
                calibration_data=self.calibration_data
            )
        elif action == "CALIBRATION":
            self.state = "CALIBRATION"
            # Reset calibration timers
            self.calib_start_time = 0
            self.calib_detector = PoseDetector()
            self.calib_samples = []
            
        elif action == "LEADERBOARD":
            self.state = "LEADERBOARD"
            
        elif action == "EXIT":
            pygame.event.post(pygame.event.Event(pygame.QUIT))

    def run_calibration_loop(self, frame):
        """
        T-pose calibration screen.
        Checks for a stable skeletal pose for 3 seconds,
        then calculates shoulder/arm and leg reference metrics.
        """
        # Mirror frame
        frame_mirrored = cv2.flip(frame, 1)
        skeleton_frame = self.calib_detector.find_pose(frame_mirrored, draw=True)
        landmarks = self.calib_detector.get_landmarks(skeleton_frame.shape)
        joints = self.calib_detector.get_key_joints(landmarks)
        
        # Render camera full screen
        frame_rgb = cv2.cvtColor(skeleton_frame, cv2.COLOR_BGR2RGB)
        frame_surf = pygame.image.frombuffer(frame_rgb.tobytes(), frame_rgb.shape[1::-1], "RGB")
        frame_scaled = pygame.transform.scale(frame_surf, (self.screen_width, self.screen_height))
        self.screen.blit(frame_scaled, (0, 0))
        
        # Render semi-transparent instructions overlay card
        instructions_panel = pygame.Surface((self.screen_width - 100, 150), pygame.SRCALPHA)
        instructions_panel.fill((15, 15, 25, 200))
        pygame.draw.rect(instructions_panel, (100, 150, 255, 128), (0, 0, self.screen_width - 100, 150), 2, border_radius=10)
        self.screen.blit(instructions_panel, (50, 40))
        
        # Render outline guide stencil
        # Draw head circle and arms cross line to guide player
        guide_color = (255, 255, 255, 50)
        pygame.draw.circle(self.screen, guide_color, (self.screen_width // 2, int(self.screen_height * 0.22)), 45, 2) # Head
        pygame.draw.line(self.screen, guide_color, (int(self.screen_width * 0.2), int(self.screen_height * 0.45)), 
                         (int(self.screen_width * 0.8), int(self.screen_height * 0.45)), 3) # Arms T
        pygame.draw.line(self.screen, guide_color, (self.screen_width // 2, int(self.screen_height * 0.22) + 45), 
                         (self.screen_width // 2, int(self.screen_height * 0.75)), 3) # Spine
        
        # Verify skeleton detection
        essential_keys = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip', 'left_ankle', 'right_ankle']
        skeleton_fully_visible = all(k in joints and joints[k]['visibility'] > 0.65 for k in essential_keys)
        
        title_font = self.assets.fonts['header']
        body_font = self.assets.fonts['body']
        
        if skeleton_fully_visible:
            # Start timer if not already running
            if self.calib_start_time == 0:
                self.calib_start_time = time.time()
                self.assets.play_sound('kick')
                
            elapsed = time.time() - self.calib_start_time
            countdown = 3.0 - elapsed
            
            # Record calibration frames
            self.calib_samples.append(joints)
            
            if countdown <= 0:
                # Calibration finished, compute values!
                self.complete_calibration()
                self.state = "MENU"
                return
                
            txt_line1 = title_font.render("HOLD T-POSE STILL...", True, (255, 230, 80))
            txt_line2 = body_font.render(f"Calibrating body proportions in {max(1, int(countdown + 1))}...", True, (255, 255, 255))
        else:
            # Reset calibration if skeleton tracking breaks
            self.calib_start_time = 0
            self.calib_samples = []
            
            txt_line1 = title_font.render("ALIGN YOUR BODY IN T-POSE", True, (255, 100, 100))
            txt_line2 = body_font.render("Ensure head, shoulders, hips, knees, and feet are fully visible.", True, (240, 240, 240))
            
        self.screen.blit(txt_line1, (80, 55))
        self.screen.blit(txt_line2, (80, 100))
        
        # Press escape to skip instruction
        skip_surf = body_font.render("Press ESC to exit calibration", True, (150, 150, 150))
        self.screen.blit(skip_surf, (self.screen_width - skip_surf.get_width() - 80, 100))

    def complete_calibration(self):
        """
        Computes physical ratios (arm span, height) from recorded samples.
        Saves calibration data.
        """
        if not self.calib_samples:
            return
            
        shoulders = []
        legs = []
        hips = []
        
        for joints in self.calib_samples:
            # 1. Shoulder Width
            ls_x = joints['left_shoulder']['x_norm']
            rs_x = joints['right_shoulder']['x_norm']
            shoulder_width = abs(ls_x - rs_x)
            shoulders.append(shoulder_width)
            
            # 2. Leg Length (hip to ankle)
            lh_y = joints['left_hip']['y_norm']
            la_y = joints['left_ankle']['y_norm']
            leg_len = la_y - lh_y
            legs.append(leg_len)
            
            # 3. Hip height reference
            hips.append(lh_y)
            
        self.calibration_data = {
            'shoulder_width': float(np.mean(shoulders)),
            'leg_length': float(np.mean(legs)),
            'hip_height': float(np.mean(hips))
        }
        self.calibrated = True
        self.assets.play_sound('goal') # Confirm sound
        print(f"Calibration successful! Metrics: {self.calibration_data}")

    def run_striker_loop(self, frame):
        """
        Striker (Mode 1) gameplay loop execution.
        """
        # Mouse/keyboard fallback trigger if camera not used
        if self.fallback_mouse_control and self.active_striker_game.state == "WAITING_KICK":
            # Scan mouse clicks to simulate a kick
            mx, my = pygame.mouse.get_pos()
            # If user clicked bottom section
            if pygame.mouse.get_pressed()[0] and my > 400:
                self.assets.play_sound('kick')
                # Determine click sector
                if mx < self.screen_width // 3:
                    direction = "LEFT"
                elif mx > 2 * self.screen_width // 3:
                    direction = "RIGHT"
                else:
                    direction = "CENTER"
                    
                # Determine click height
                if my < 550:
                    height = "HIGH"
                elif my > 680:
                    height = "LOW"
                else:
                    height = "MID"
                    
                # Setup fake kick event
                self.active_striker_game.kick_info = {
                    'leg': 'right',
                    'velocity': 0.08,
                    'power': random.randint(50, 95),
                    'direction': direction,
                    'height': height,
                    'kick_angle': float(random.randint(-15, 15)),
                    'body_orientation': 0.0,
                    'prediction': direction,
                    'confidence': 1.0
                }
                # Load physics
                self.active_striker_game.ball_trajectory = self.active_striker_game.physics.calculate_trajectory(
                    direction, height, self.active_striker_game.kick_info['power'], self.active_striker_game.kick_info['kick_angle']
                )
                self.active_striker_game.trajectory_index = 0
                self.active_striker_game.determine_ai_goalkeeper_dive(direction, height)
                self.active_striker_game.state = "BALL_FLIGHT"

        # Update gameplay
        skeleton_frame = self.active_striker_game.update(frame)
        self.active_striker_game.draw(skeleton_frame)
        
        # State machine transition to Replay or End
        if self.active_striker_game.state == "COMPLETED":
            # Store attempt data for replay
            self.replay_buffer = self.active_striker_game.replay_data.copy()
            self.replay_mode_type = "STRIKER"
            
            # Transition
            self.state = "REPLAY"
            self.replay_frame_index = 0

    def run_keeper_loop(self, frame):
        """
        Goalkeeper (Mode 2) gameplay loop execution.
        """
        # Mouse fallback control: override hand positions with mouse coordinate
        if self.fallback_mouse_control:
            mx, my = pygame.mouse.get_pos()
            self.active_keeper_game.left_glove_pos = (mx - 60, my)
            self.active_keeper_game.right_glove_pos = (mx + 60, my)
            
        skeleton_frame = self.active_keeper_game.update(frame)
        self.active_keeper_game.draw(skeleton_frame)
        
        # Override gloves drawing in fallback
        if self.fallback_mouse_control:
            self.active_keeper_game._draw_gloves()

        # State transition to Replay or End
        if self.active_keeper_game.state == "COMPLETED":
            # Save replay
            self.replay_buffer = self.active_keeper_game.replay_data.copy()
            self.replay_mode_type = "GOALKEEPER"
            
            # Transition to replay first
            self.state = "REPLAY"
            self.replay_frame_index = 0

    def run_replay_loop(self):
        """
        Cinematic slow-motion replay of the last penalty attempt.
        """
        # Render background stadium
        self.screen.blit(self.assets.images['stadium'], (0, 0))
        
        # Net boundary lines
        if self.assets.images['goal']:
            self.screen.blit(self.assets.images['goal'], (self.physics.goal_left, self.physics.goal_top))
        else:
            gl = self.physics.goal_left
            gr = self.physics.goal_right
            gt = self.physics.goal_top
            gb = self.physics.goal_bottom
            pygame.draw.line(self.screen, (240, 240, 240), (gl, gb), (gl, gt), 8)
            pygame.draw.line(self.screen, (240, 240, 240), (gr, gb), (gr, gt), 8)
            pygame.draw.line(self.screen, (240, 240, 240), (gl - 4, gt), (gr + 4, gt), 8)
            
        # Draw frame from replay buffer
        if self.replay_buffer and self.replay_frame_index < len(self.replay_buffer):
            data = self.replay_buffer[self.replay_frame_index]
            
            # 1. Draw skeleton of recorded frame (Neon pink color)
            self.draw_replay_skeleton(data['landmarks'])
            
            # 2. Draw gloves (if Goalkeeper Mode) or goalie avatar (if Striker Mode)
            if self.replay_mode_type == "STRIKER":
                # Render goalie diving
                keeper_img = self.assets.images['goalkeeper']
                rot = data['keeper_rotation']
                if rot != 0:
                    rotated_keeper = pygame.transform.rotate(keeper_img, rot)
                else:
                    rotated_keeper = keeper_img
                k_pos = data['keeper_pos']
                k_rect = rotated_keeper.get_rect(center=k_pos)
                self.screen.blit(rotated_keeper, k_rect.topleft)
            else:
                # Render gloves
                lg = data['left_glove']
                rg = data['right_glove']
                pygame.draw.circle(self.screen, (220, 50, 50), lg, 45)
                pygame.draw.circle(self.screen, (50, 100, 220), rg, 45)
                
            # 3. Draw Ball
            ball_img = self.assets.images['ball']
            bx, by = data['ball_pos']
            br = data['ball_radius']
            scaled_ball = pygame.transform.scale(ball_img, (br * 2, br * 2))
            ball_rect = scaled_ball.get_rect(center=(bx, by))
            self.screen.blit(scaled_ball, ball_rect.topleft)
            
            # Progress index slowly (every 2nd tick is standard slow-motion)
            if pygame.time.get_ticks() % 2 == 0:
                self.replay_frame_index += 1
        else:
            # Replay complete: resume next kick/attempt or handle game over
            if self.replay_mode_type == "STRIKER":
                if self.scores.is_game_over():
                    self.qualifying_score = self.scores.goals
                    self.leaderboard_mode = "STRIKER"
                    self.check_leaderboard_qualification()
                else:
                    self.active_striker_game.reset_attempt()
                    self.state = "PLAY_STRIKER"
            else:
                if self.scores.is_game_over():
                    self.qualifying_score = self.scores.saves
                    self.leaderboard_mode = "GOALKEEPER"
                    self.check_leaderboard_qualification()
                else:
                    self.active_keeper_game.reset_attempt()
                    self.state = "PLAY_GOALKEEPER"
                
        # Draw Replay text
        hud_panel = pygame.Surface((300, 55), pygame.SRCALPHA)
        hud_panel.fill((15, 15, 25, 220))
        pygame.draw.rect(hud_panel, (255, 50, 150), (0, 0, 300, 55), 2, border_radius=8)
        self.screen.blit(hud_panel, (20, 20))
        
        # Blinking replay label
        alpha = int(127 * np.sin(time.time() * 6) + 128)
        rep_color = (255, 50, 150, alpha)
        
        rep_lbl = self.assets.fonts['hud'].render("• SLOW-MOTION REPLAY", True, rep_color)
        self.screen.blit(rep_lbl, (35, 36))
        
        # Draw instruction to skip
        skip_lbl = self.assets.fonts['body'].render("Press SPACE to skip replay", True, (200, 200, 200))
        self.screen.blit(skip_lbl, (self.screen_width - skip_lbl.get_width() - 40, 30))
        
        # Check skip trigger
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            self.replay_frame_index = len(self.replay_buffer) # Skip to end

    def draw_replay_skeleton(self, landmarks):
        """
        Draws a neon pink skeletal overlay from recorded frame landmarks on the replay screen.
        """
        if not landmarks:
            return
            
        pose_mp = PoseDetector()
        connections = pose_mp.mp_pose.POSE_CONNECTIONS
        
        shoulder_width = self.calibration_data.get('shoulder_width', 0.15)
        arm_span = shoulder_width * 2.2
        
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx not in landmarks or end_idx not in landmarks:
                continue
                
            lm_start = landmarks[start_idx]
            lm_end = landmarks[end_idx]
            
            if lm_start['visibility'] < 0.4 or lm_end['visibility'] < 0.4:
                continue
                
            # If striker mode, map skeleton to camera box space in corner or center
            # Let's map it centered in the bottom of the screen
            if self.replay_mode_type == "STRIKER":
                # Bottom overlay
                sx = int(self.screen_width * 0.15 + lm_start['x_norm'] * 200)
                sy = int(self.screen_height * 0.6 + lm_start['y_norm'] * 200)
                ex = int(self.screen_width * 0.15 + lm_end['x_norm'] * 200)
                ey = int(self.screen_height * 0.6 + lm_end['y_norm'] * 200)
            else:
                # Goalkeeper Mode connection: map onto goal line
                lh = landmarks[pose_mp.mp_pose.PoseLandmark.LEFT_HIP.value]
                rh = landmarks[pose_mp.mp_pose.PoseLandmark.RIGHT_HIP.value]
                body_center_x = (lh['x_norm'] + rh['x_norm']) / 2.0
                body_y = (lh['y_norm'] + rh['y_norm']) / 2.0
                
                dx_s = lm_start['x_norm'] - body_center_x
                dy_s = body_y - lm_start['y_norm']
                
                sens_x = 1.3 / arm_span
                sens_y = 1.4 / arm_span
                
                sx = int(self.physics.goal_center_x + dx_s * sens_x * self.physics.goal_width)
                sy = int(self.physics.goal_bottom - (dy_s + 0.1) * sens_y * self.physics.goal_height)
                
                dx_e = lm_end['x_norm'] - body_center_x
                dy_e = body_y - lm_end['y_norm']
                
                ex = int(self.physics.goal_center_x + dx_e * sens_x * self.physics.goal_width)
                ey = int(self.physics.goal_bottom - (dy_e + 0.1) * sens_y * self.physics.goal_height)
                
            pygame.draw.line(self.screen, (255, 50, 150), (sx, sy), (ex, ey), 3)

    def check_leaderboard_qualification(self):
        """
        Determines if the score is eligible to enter the Top 10.
        If yes, moves to NAME_INPUT state, otherwise routes directly to LEADERBOARD.
        """
        top_scores = self.scores.get_top_scores(self.leaderboard_mode)
        
        # Qualified if less than 10 scores recorded, or score is higher than the lowest highscore
        qualified = len(top_scores) < 10 or self.qualifying_score > top_scores[-1]["score"]
        
        if qualified:
            self.player_name = ""
            self.state = "NAME_INPUT"
        else:
            self.state = "LEADERBOARD"

    def handle_name_input_events(self, event):
        """
        Text editor inputs for the name registry.
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                if len(self.player_name.strip()) > 0:
                    name = self.player_name.strip().upper()
                    diff = self.menu.get_difficulty()
                    self.scores.add_score(name, self.leaderboard_mode, self.qualifying_score, diff)
                    self.assets.play_sound('goal')
                    self.state = "LEADERBOARD"
            elif event.key == pygame.K_BACKSPACE:
                self.player_name = self.player_name[:-1]
            else:
                # Limit name size to 8 characters alphanumeric
                if len(self.player_name) < 8 and event.unicode.isalnum():
                    self.player_name += event.unicode.upper()

    def draw_name_input_screen(self):
        """
        Name prompt visual layout.
        """
        self.screen.blit(self.assets.images['stadium'], (0, 0))
        
        # Center panel
        panel_rect = pygame.Rect(self.screen_width // 2 - 250, self.screen_height // 2 - 160, 500, 320)
        self.menu.draw_glass_panel(panel_rect, color=(20, 20, 30), alpha=230)
        
        title_font = self.assets.fonts['header']
        body_font = self.assets.fonts['body']
        
        # Title
        t_surf = title_font.render("NEW HIGH SCORE!", True, (255, 230, 80))
        self.screen.blit(t_surf, (panel_rect.x + (panel_rect.width - t_surf.get_width()) // 2, panel_rect.y + 30))
        
        # Score message
        mode_str = "Striker" if self.leaderboard_mode == "STRIKER" else "Goalkeeper"
        metric = "Goals" if self.leaderboard_mode == "STRIKER" else "Saves"
        score_text = f"You scored {self.qualifying_score} {metric} on {self.menu.get_difficulty()}!"
        s_surf = body_font.render(score_text, True, (255, 255, 255))
        self.screen.blit(s_surf, (panel_rect.x + (panel_rect.width - s_surf.get_width()) // 2, panel_rect.y + 90))
        
        # Input label
        prompt = body_font.render("ENTER NAME (8 CHARS MAX):", True, (100, 180, 255))
        self.screen.blit(prompt, (panel_rect.x + (panel_rect.width - prompt.get_width()) // 2, panel_rect.y + 140))
        
        # Text Box
        box_rect = pygame.Rect(panel_rect.x + 100, panel_rect.y + 180, 300, 50)
        pygame.draw.rect(self.screen, (10, 10, 15), box_rect, border_radius=8)
        pygame.draw.rect(self.screen, (50, 120, 255), box_rect, 2, border_radius=8)
        
        # Text drawing (add cursor blinking effect)
        cursor = "_" if int(time.time() * 2) % 2 == 0 else ""
        name_surf = title_font.render(f"{self.player_name}{cursor}", True, (255, 255, 255))
        self.screen.blit(name_surf, (box_rect.x + (box_rect.width - name_surf.get_width()) // 2, box_rect.y + 5))
        
        # Confirm notice
        notice = body_font.render("PRESS ENTER TO REGISTER", True, (150, 150, 150))
        self.screen.blit(notice, (panel_rect.x + (panel_rect.width - notice.get_width()) // 2, panel_rect.y + 255))

    def draw_leaderboard_screen(self):
        """
        Visualizes high scores table.
        """
        self.screen.blit(self.assets.images['stadium'], (0, 0))
        
        panel_rect = pygame.Rect(100, 60, self.screen_width - 200, self.screen_height - 120)
        self.menu.draw_glass_panel(panel_rect, color=(15, 15, 25), alpha=220)
        
        title_font = self.assets.fonts['title']
        header_font = self.assets.fonts['header']
        body_font = self.assets.fonts['body']
        
        # Title
        t_surf = title_font.render("LOCAL LEADERBOARD", True, (50, 200, 255))
        self.screen.blit(t_surf, (self.screen_width // 2 - t_surf.get_width() // 2, 85))
        
        # Headers
        h_rank = header_font.render("RK", True, (255, 230, 80))
        h_name = header_font.render("NAME", True, (255, 230, 80))
        h_mode = header_font.render("MODE", True, (255, 230, 80))
        h_score = header_font.render("SCORE", True, (255, 230, 80))
        h_diff = header_font.render("DIFFICULTY", True, (255, 230, 80))
        
        start_y = 190
        self.screen.blit(h_rank, (150, start_y))
        self.screen.blit(h_name, (220, start_y))
        self.screen.blit(h_mode, (380, start_y))
        self.screen.blit(h_score, (580, start_y))
        self.screen.blit(h_diff, (720, start_y))
        
        # Draw line under headers
        pygame.draw.line(self.screen, (50, 120, 255, 100), (140, start_y + 40), (self.screen_width - 140, start_y + 40), 2)
        
        # Fetch Top 10
        records = self.scores.get_top_scores()[:10]
        
        row_y = start_y + 50
        row_height = 42
        
        for idx, rec in enumerate(records):
            color = (255, 255, 255)
            # Golden highlight for 1st place
            if idx == 0:
                color = (255, 215, 0)
            elif idx == 1:
                color = (200, 200, 200) # Silver
            elif idx == 2:
                color = (205, 127, 50) # Bronze
                
            r_surf = body_font.render(f"{idx + 1:02d}", True, color)
            n_surf = body_font.render(rec['name'], True, color)
            
            mode_lbl = "STRIKER" if rec['mode'] == "STRIKER" else "GK MODE"
            m_surf = body_font.render(mode_lbl, True, color)
            
            metric = "Goals" if rec['mode'] == "STRIKER" else "Saves"
            s_surf = body_font.render(f"{rec['score']} {metric}", True, color)
            d_surf = body_font.render(rec['difficulty'], True, color)
            
            self.screen.blit(r_surf, (150, row_y + idx * row_height))
            self.screen.blit(n_surf, (220, row_y + idx * row_height))
            self.screen.blit(m_surf, (380, row_y + idx * row_height))
            self.screen.blit(s_surf, (580, row_y + idx * row_height))
            self.screen.blit(d_surf, (720, row_y + idx * row_height))
            
        # Exit notice
        ex_surf = body_font.render("PRESS ANY KEY TO RETURN TO MAIN MENU", True, (150, 150, 150))
        self.screen.blit(ex_surf, (self.screen_width // 2 - ex_surf.get_width() // 2, self.screen_height - 110))

    def draw_fps(self):
        """
        Draws current frame rate and device controls status.
        """
        fps = int(self.clock.get_fps())
        cam_status = "WEBCAM: ON" if self.webcam_connected else "WEBCAM: MOCK (MOUSE)"
        cal_status = "CALIBRATION: ACTIVE" if self.calibrated else "CALIBRATION: DEFAULT"
        
        info_str = f"FPS: {fps} | {cam_status} | {cal_status}"
        fps_surf = self.fps_font.render(info_str, True, (0, 255, 50))
        
        # Transparent background tag
        tag_surf = pygame.Surface((fps_surf.get_width() + 16, 25), pygame.SRCALPHA)
        tag_surf.fill((10, 10, 10, 180))
        
        self.screen.blit(tag_surf, (10, self.screen_height - 35))
        self.screen.blit(fps_surf, (18, self.screen_height - 31))

if __name__ == "__main__":
    game = GameOrchestrator()
    game.run()
