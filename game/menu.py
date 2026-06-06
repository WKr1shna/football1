import pygame
import cv2
import sys

class MainMenu:
    """
    Main Menu UI for the game.
    Displays options: Penalty Striker, Goalkeeper Mode, Difficulty, Calibration, Leaderboard, Exit.
    Shows the camera feed in a corner box to help the player align themselves.
    """
    def __init__(self, screen, asset_manager, score_manager, cap):
        self.screen = screen
        self.assets = asset_manager
        self.scores = score_manager
        self.cap = cap
        
        self.width = screen.get_width()
        self.height = screen.get_height()
        
        # Menu options
        self.options = [
            "STRIKER MODE",
            "GOALKEEPER MODE",
            "DIFFICULTY: MEDIUM",
            "CAMERA CALIBRATION",
            "LEADERBOARD",
            "EXIT"
        ]
        self.selected_index = 0
        
        # Difficulties
        self.difficulties = ["EASY", "MEDIUM", "HARD"]
        self.difficulty_index = 1 # Medium default
        
        # Layout metrics
        self.panel_rect = pygame.Rect(50, 50, 480, self.height - 100)
        self.webcam_rect = pygame.Rect(self.width - 400, 50, 350, 262) # 4:3 ratio scaled

    def get_difficulty(self):
        return self.difficulties[self.difficulty_index]

    def update_difficulty_text(self):
        self.options[2] = f"DIFFICULTY: {self.get_difficulty()}"

    def draw_glass_panel(self, rect, color=(20, 20, 35), alpha=190):
        """
        Draws a gorgeous semi-transparent card overlay with a subtle border glow.
        """
        glass_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        glass_surf.fill((*color, alpha))
        
        # Border
        pygame.draw.rect(glass_surf, (100, 150, 255, 100), (0, 0, rect.width, rect.height), 2, border_radius=15)
        self.screen.blit(glass_surf, rect.topleft)

    def draw(self, frame_raw):
        """
        Renders the entire menu page.
        """
        # 1. Draw stadium background
        self.screen.blit(self.assets.images['stadium'], (0, 0))
        
        # 2. Draw Left Glass Menu Panel
        self.draw_glass_panel(self.panel_rect)
        
        # 3. Draw Title
        title_surf = self.assets.fonts['title'].render("VISION", True, (255, 255, 255))
        title_surf2 = self.assets.fonts['title'].render("FOOTBALL", True, (50, 200, 255))
        self.screen.blit(title_surf, (80, 80))
        self.screen.blit(title_surf2, (80, 140))
        
        # 4. Draw Menu Options
        start_y = 230
        spacing = 55
        
        for idx, option in enumerate(self.options):
            is_selected = (idx == self.selected_index)
            color = (255, 255, 255) if not is_selected else (255, 230, 80)
            prefix = "> " if is_selected else "  "
            
            # Hover background for selected item
            if is_selected:
                hover_rect = pygame.Rect(70, start_y + idx * spacing - 5, 440, 42)
                hover_surf = pygame.Surface((hover_rect.width, hover_rect.height), pygame.SRCALPHA)
                hover_surf.fill((255, 255, 255, 25))
                pygame.draw.rect(hover_surf, (255, 230, 80, 80), (0, 0, hover_rect.width, hover_rect.height), 1, border_radius=5)
                self.screen.blit(hover_surf, hover_rect.topleft)
                
            text_surf = self.assets.fonts['header'].render(f"{prefix}{option}", True, color)
            self.screen.blit(text_surf, (80, start_y + idx * spacing))
            
        # 5. Draw Webcam Corner Preview
        self.draw_glass_panel(self.webcam_rect)
        if frame_raw is not None:
            # Mirror frame for intuitive self-reflection
            frame_mirrored = cv2.flip(frame_raw, 1)
            # Convert OpenCV frame (BGR) to Pygame surface
            frame_rgb = cv2.cvtColor(frame_mirrored, cv2.COLOR_BGR2RGB)
            frame_surf = pygame.image.frombuffer(frame_rgb.tobytes(), frame_rgb.shape[1::-1], "RGB")
            frame_scaled = pygame.transform.scale(frame_surf, (self.webcam_rect.width - 6, self.webcam_rect.height - 6))
            self.screen.blit(frame_scaled, (self.webcam_rect.x + 3, self.webcam_rect.y + 3))
            
            # Label overlay
            preview_lbl = self.assets.fonts['body'].render("LIVE CAMERA PREVIEW", True, (100, 255, 150))
            self.screen.blit(preview_lbl, (self.webcam_rect.x + 10, self.webcam_rect.y + self.webcam_rect.height + 10))
            
        # 6. Draw game instructions card at the bottom right
        info_rect = pygame.Rect(self.width - 400, 370, 350, self.height - 420)
        self.draw_glass_panel(info_rect, color=(15, 15, 25), alpha=210)
        
        info_title = self.assets.fonts['header'].render("HOW TO PLAY", True, (255, 255, 255))
        self.screen.blit(info_title, (info_rect.x + 20, info_rect.y + 15))
        
        instructions = [
            "1. Run CAMERA CALIBRATION first",
            "   to calibrate body detection.",
            "",
            "2. STRIKER MODE:",
            "   Step back so full body is visible.",
            "   Kick foot left/right/center.",
            "",
            "3. GOALKEEPER MODE:",
            "   Use hands to block the ball",
            "   on screen path."
        ]
        
        for idx, line in enumerate(instructions):
            line_surf = self.assets.fonts['body'].render(line, True, (200, 200, 220))
            self.screen.blit(line_surf, (info_rect.x + 20, info_rect.y + 55 + idx * 22))

    def handle_event(self, event):
        """
        Process user input to navigate.
        Returns the action string if selected, else None.
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.selected_index = (self.selected_index - 1) % len(self.options)
                self.assets.play_sound('kick') # Subtle click sound
            elif event.key == pygame.K_DOWN:
                self.selected_index = (self.selected_index + 1) % len(self.options)
                self.assets.play_sound('kick')
            elif event.key == pygame.K_RETURN:
                self.assets.play_sound('save') # Affirmative sound
                return self.execute_selection()
                
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Check options clicking
            start_y = 230
            spacing = 55
            mx, my = pygame.mouse.get_pos()
            
            # Check if clicked left panel menu area
            if self.panel_rect.collidepoint(mx, my):
                for idx in range(len(self.options)):
                    opt_rect = pygame.Rect(70, start_y + idx * spacing - 5, 440, 42)
                    if opt_rect.collidepoint(mx, my):
                        self.selected_index = idx
                        self.assets.play_sound('save')
                        return self.execute_selection()
                        
        return None

    def execute_selection(self):
        """
        Returns action string.
        """
        if self.selected_index == 0:
            return "STRIKER"
        elif self.selected_index == 1:
            return "GOALKEEPER"
        elif self.selected_index == 2:
            # Toggle difficulty
            self.difficulty_index = (self.difficulty_index + 1) % len(self.difficulties)
            self.update_difficulty_text()
            return f"DIFF_{self.get_difficulty()}"
        elif self.selected_index == 3:
            return "CALIBRATION"
        elif self.selected_index == 4:
            return "LEADERBOARD"
        elif self.selected_index == 5:
            return "EXIT"
        return None
