import pygame
import os
import numpy as np

class AssetManager:
    """
    Manages loading of images, fonts, and sound effects.
    If files are missing, it programmatically generates stunning vector textures 
    and synthesizes high-quality audio sound effects from raw NumPy sine/noise arrays.
    """
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.assets_dir = os.path.join(current_dir, "..", "assets")
        self.sounds_dir = os.path.join(self.assets_dir, "sounds")
        
        # Audio setup
        pygame.mixer.init()
        
        # Load or generate assets
        self.images = {}
        self.sounds = {}
        self.fonts = {}
        
        self.load_fonts()
        self.load_or_create_images()
        self.load_or_synthesize_sounds()

    def load_fonts(self):
        """
        Loads standard system fonts or default fallback fonts.
        """
        try:
            self.fonts['title'] = pygame.font.SysFont("Outfit", 64, bold=True)
            self.fonts['header'] = pygame.font.SysFont("Outfit", 36, bold=True)
            self.fonts['body'] = pygame.font.SysFont("Inter", 24)
            self.fonts['hud'] = pygame.font.SysFont("monospace", 22, bold=True)
        except Exception:
            # Fallback to pygame default font
            self.fonts['title'] = pygame.font.Font(None, 64)
            self.fonts['header'] = pygame.font.Font(None, 36)
            self.fonts['body'] = pygame.font.Font(None, 24)
            self.fonts['hud'] = pygame.font.Font(None, 22)

    def load_or_create_images(self):
        """
        Attempts to load PNG/JPG assets. If missing, draws them procedurally.
        """
        # 1. Stadium Background
        stadium_path = os.path.join(self.assets_dir, "stadium.png")
        if not os.path.exists(stadium_path):
            stadium_path = os.path.join(self.assets_dir, "stadium_bg.jpg")
            
        if os.path.exists(stadium_path):
            try:
                bg = pygame.image.load(stadium_path).convert()
                self.images['stadium'] = pygame.transform.scale(bg, (self.screen_width, self.screen_height))
            except Exception as e:
                print(f"Failed to load stadium image: {e}")
                self.images['stadium'] = self._generate_procedural_stadium()
        else:
            self.images['stadium'] = self._generate_procedural_stadium()
 
        # 2. Football
        ball_path = os.path.join(self.assets_dir, "football.png")
        if not os.path.exists(ball_path):
            ball_path = os.path.join(self.assets_dir, "ball.png")
            
        if os.path.exists(ball_path):
            try:
                self.images['ball'] = pygame.image.load(ball_path).convert_alpha()
            except Exception as e:
                print(f"Failed to load ball image: {e}")
                self.images['ball'] = self._generate_procedural_ball()
        else:
            self.images['ball'] = self._generate_procedural_ball()
 
        # 3. Goalkeeper
        goalkeeper_path = os.path.join(self.assets_dir, "goalkeeper_idle.png")
        if not os.path.exists(goalkeeper_path):
            goalkeeper_path = os.path.join(self.assets_dir, "goalkeeper.png")
            
        if os.path.exists(goalkeeper_path):
            try:
                gk_img = pygame.image.load(goalkeeper_path).convert_alpha()
                self.images['goalkeeper'] = pygame.transform.scale(gk_img, (130, 170))
            except Exception as e:
                print(f"Failed to load goalkeeper image: {e}")
                self.images['goalkeeper'] = pygame.transform.scale(self._generate_procedural_goalkeeper(), (130, 170))
        else:
            self.images['goalkeeper'] = pygame.transform.scale(self._generate_procedural_goalkeeper(), (130, 170))

        # 4. Goal
        goal_path = os.path.join(self.assets_dir, "goal.png")
        goal_w = int(self.screen_width * 0.45)
        goal_h = int(self.screen_height * 0.35)
        if os.path.exists(goal_path):
            try:
                goal_img = pygame.image.load(goal_path).convert_alpha()
                self.images['goal'] = pygame.transform.scale(goal_img, (goal_w, goal_h))
            except Exception as e:
                print(f"Failed to load goal image: {e}")
                self.images['goal'] = None # If None, code will draw it using Pygame vectors
        else:
            self.images['goal'] = None

    def load_or_synthesize_sounds(self):
        """
        Attempts to load WAV sounds. If missing, synthesizes them from mathematical waveforms.
        """
        sound_files = {
            'kick': 'kick.wav',
            'save': 'save.wav',
            'goal': 'goal.wav',
            'miss': 'miss.wav',
            'cheer': 'cheer.wav'
        }
        
        for key, filename in sound_files.items():
            path = os.path.join(self.sounds_dir, filename)
            if os.path.exists(path):
                try:
                    self.sounds[key] = pygame.mixer.Sound(path)
                    continue
                except Exception as e:
                    print(f"Error loading sound {filename}: {e}")
            
            # Synthesize fallback sound
            self.sounds[key] = self._synthesize_sound(key)

    def _synthesize_sound(self, key, sample_rate=44100):
        """
        Generates 16-bit PCM sound effects using NumPy formulas.
        """
        duration = 0.5
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        if key == 'kick':
            # Low pitch thud: pitch slides down rapidly (150Hz -> 30Hz)
            duration = 0.15
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            freqs = 150.0 * np.exp(-25.0 * t) + 30.0
            phase = 2.0 * np.pi * np.cumsum(freqs) / sample_rate
            wave = np.sin(phase)
            envelope = np.exp(-12.0 * t)
            wave = wave * envelope
            
        elif key == 'save':
            # Punchy slap: white noise + quick tone (300Hz -> 100Hz)
            duration = 0.2
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            freqs = 300.0 * np.exp(-15.0 * t) + 100.0
            phase = 2.0 * np.pi * np.cumsum(freqs) / sample_rate
            tone = np.sin(phase)
            noise = np.random.uniform(-1.0, 1.0, len(t))
            wave = 0.4 * tone + 0.6 * noise
            envelope = np.exp(-18.0 * t)
            wave = wave * envelope
            
        elif key == 'goal':
            # Referee whistle: dual high-frequency tone producing beats (2000Hz + 2030Hz)
            duration = 0.6
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            tone1 = np.sin(2 * np.pi * 2000.0 * t)
            tone2 = np.sin(2 * np.pi * 2030.0 * t)
            wave = 0.5 * (tone1 + tone2)
            # Add vibrato effect
            vibrato = np.sin(2 * np.pi * 50.0 * t) * 0.1 + 0.9
            # Envelope (quick attack, slight fade)
            envelope = np.clip(t / 0.05, 0, 1) * np.exp(-2.0 * t)
            wave = wave * envelope * vibrato
            
        elif key == 'miss':
            # Descending disappointed slide
            duration = 0.5
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            freqs = 220.0 * np.exp(-3.0 * t)
            phase = 2.0 * np.pi * np.cumsum(freqs) / sample_rate
            wave = np.sin(phase)
            envelope = np.exp(-3.0 * t)
            wave = wave * envelope
            
        elif key == 'cheer':
            # Stadium crowd cheer: layered white noise with slow attack and decay
            duration = 2.0
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            # Generate pseudo-white noise
            noise = np.random.uniform(-1.0, 1.0, len(t))
            # Smooth/lowpass the noise a bit by rolling average to make it sound like a crowd
            noise_smoothed = np.convolve(noise, np.ones(20)/20, mode='same')
            # Slow rise, long decay envelope
            envelope = np.minimum(t / 0.4, 1.0) * np.exp(-1.5 * (t - 0.4) * (t > 0.4))
            wave = noise_smoothed * envelope * 0.7
            
        else:
            wave = np.sin(2 * np.pi * 440.0 * t) * 0.1
            
        # Convert to 16-bit signed PCM
        audio = (wave * 32767).astype(np.int16)
        
        # Convert to 2-channel stereo for Pygame mixer
        stereo = np.column_stack((audio, audio))
        
        try:
            return pygame.sndarray.make_sound(stereo)
        except Exception as e:
            print(f"Failed to synthesize sound '{key}': {e}")
            # Return empty sound if mixer not working
            return pygame.mixer.Sound(buffer=bytes(100))

    def _generate_procedural_stadium(self):
        """
        Creates a beautiful stadium background surface with sunset gradient and pitch lines.
        """
        surface = pygame.Surface((self.screen_width, self.screen_height))
        
        # 1. Sky Sunset Gradient (Top 45% of screen)
        sky_height = int(self.screen_height * 0.45)
        for y in range(sky_height):
            # Gradient from deep purple (40, 20, 70) to warm orange (240, 100, 50)
            ratio = y / sky_height
            r = int(40 * (1 - ratio) + 240 * ratio)
            g = int(20 * (1 - ratio) + 100 * ratio)
            b = int(70 * (1 - ratio) + 50 * ratio)
            pygame.draw.line(surface, (r, g, b), (0, y), (self.screen_width, y))
            
        # Draw some soft yellow stadium floodlight flares
        lights_y = int(self.screen_height * 0.1)
        pygame.draw.circle(surface, (255, 255, 200, 128), (int(self.screen_width * 0.15), lights_y), 45)
        pygame.draw.circle(surface, (255, 255, 200, 128), (int(self.screen_width * 0.85), lights_y), 45)
        
        # 2. Football Pitch Gradient (Bottom 55% of screen)
        pitch_y = sky_height
        pitch_height = self.screen_height - pitch_y
        for y in range(pitch_y, self.screen_height):
            ratio = (y - pitch_y) / pitch_height
            # Gradient from darker green (10, 80, 20) to lighter field green (30, 150, 40)
            r = int(10 * (1 - ratio) + 30 * ratio)
            g = int(80 * (1 - ratio) + 160 * ratio)
            b = int(20 * (1 - ratio) + 40 * ratio)
            pygame.draw.line(surface, (r, g, b), (0, y), (self.screen_width, y))
            
        # Draw perspective grass stripes (alternating greens)
        num_stripes = 8
        for i in range(num_stripes):
            stripe_top = pitch_y + (i * pitch_height // num_stripes)
            stripe_bot = pitch_y + ((i + 1) * pitch_height // num_stripes)
            if i % 2 == 0:
                stripe_surf = pygame.Surface((self.screen_width, stripe_bot - stripe_top), pygame.SRCALPHA)
                stripe_surf.fill((255, 255, 255, 12)) # soft overlay
                surface.blit(stripe_surf, (0, stripe_top))
                
        # Draw penalty box in perspective
        # Goal line is at y = pitch_y
        # Front of box is at y = int(self.screen_height * 0.75)
        box_top_y = pitch_y
        box_bot_y = int(self.screen_height * 0.70)
        
        gl = int(self.screen_width * 0.28)
        gr = int(self.screen_width * 0.72)
        bl = int(self.screen_width * 0.12)
        br = int(self.screen_width * 0.88)
        
        # Draw white line markers
        pygame.draw.line(surface, (230, 230, 230), (gl, box_top_y), (bl, box_bot_y), 3) # left line
        pygame.draw.line(surface, (230, 230, 230), (gr, box_top_y), (br, box_bot_y), 3) # right line
        pygame.draw.line(surface, (230, 230, 230), (bl, box_bot_y), (br, box_bot_y), 3) # front line
        
        # Penalty spot (Z=0 point for physics)
        pygame.draw.circle(surface, (230, 230, 230), (self.screen_width // 2, int(self.screen_height * 0.82)), 6)
        
        # Stand outlines (background stadium structures)
        # Draw a dark gray silhouette representing the stadium stands above the pitch line
        pygame.draw.polygon(surface, (30, 30, 45), [
            (0, pitch_y),
            (0, int(pitch_y * 0.7)),
            (int(self.screen_width * 0.2), int(pitch_y * 0.8)),
            (int(self.screen_width * 0.8), int(pitch_y * 0.8)),
            (self.screen_width, int(pitch_y * 0.7)),
            (self.screen_width, pitch_y)
        ])
        
        return surface

    def _generate_procedural_ball(self):
        """
        Creates a high-quality 2D football sprite using vector math.
        """
        size = 80
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        center = size // 2
        radius = size // 2 - 2
        
        # 1. Base sphere shadow and glow
        pygame.draw.circle(surface, (20, 20, 20, 80), (center + 2, center + 2), radius) # Shadow
        pygame.draw.circle(surface, (245, 245, 245), (center, center), radius) # Base white ball
        
        # 2. Draw black pentagonal patches
        # Center patch
        pent_pts_center = []
        for i in range(5):
            angle = np.radians(i * 72 - 18)
            x = center + int(12 * np.cos(angle))
            y = center + int(12 * np.sin(angle))
            pent_pts_center.append((x, y))
        pygame.draw.polygon(surface, (35, 35, 35), pent_pts_center)
        
        # Outer patches (edges)
        for rot in range(5):
            rot_angle = np.radians(rot * 72 - 18)
            cx = center + int(26 * np.cos(rot_angle))
            cy = center + int(26 * np.sin(rot_angle))
            
            # Draw lines connecting center patch to outer patches (seams)
            c_pt = pent_pts_center[rot]
            pygame.draw.line(surface, (40, 40, 40), c_pt, (cx, cy), 2)
            
            # Outer small partial pentagons
            outer_pts = []
            for i in range(5):
                angle = np.radians(i * 72 - 18 + rot * 72)
                x = cx + int(8 * np.cos(angle))
                y = cy + int(8 * np.sin(angle))
                outer_pts.append((x, y))
            pygame.draw.polygon(surface, (35, 35, 35), outer_pts)
            
        # Draw seams (outer border circle)
        pygame.draw.circle(surface, (40, 40, 40), (center, center), radius, 2)
        
        # Shine highlight overlay
        shine = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.circle(shine, (255, 255, 255, 80), (center - 8, center - 8), radius // 2)
        surface.blit(shine, (0, 0))
        
        return surface

    def _generate_procedural_goalkeeper(self):
        """
        Creates a stylized goalkeeper placeholder image (standing ready).
        """
        w, h = 120, 160
        surface = pygame.Surface((w, h), pygame.SRCALPHA)
        
        # Draw silhouette of goalkeeper
        # Head
        pygame.draw.circle(surface, (240, 200, 160), (w // 2, 25), 18)
        # Cap/Hair
        pygame.draw.arc(surface, (30, 30, 30), (w // 2 - 18, 5, 36, 30), 0, np.pi, 5)
        
        # Body (Jersey - Cyan/Yellow stripes)
        pygame.draw.polygon(surface, (230, 200, 10), [(w // 2 - 25, 45), (w // 2 + 25, 45), (w // 2 + 20, 105), (w // 2 - 20, 105)])
        # Striping details
        pygame.draw.line(surface, (20, 120, 220), (w // 2 - 10, 45), (w // 2 - 8, 105), 8)
        pygame.draw.line(surface, (20, 120, 220), (w // 2 + 10, 45), (w // 2 + 8, 105), 8)
        
        # Arms (stretched out horizontally/upwards)
        # Left arm
        pygame.draw.line(surface, (230, 200, 10), (w // 2 - 25, 50), (w // 2 - 60, 35), 14) # jersey sleeve
        pygame.draw.line(surface, (240, 200, 160), (w // 2 - 55, 37), (w // 2 - 90, 25), 10) # bare arm
        pygame.draw.circle(surface, (220, 30, 30), (w // 2 - 95, 22), 12) # red glove
        
        # Right arm
        pygame.draw.line(surface, (230, 200, 10), (w // 2 + 25, 50), (w // 2 + 60, 35), 14) # jersey sleeve
        pygame.draw.line(surface, (240, 200, 160), (w // 2 + 55, 37), (w // 2 + 90, 25), 10) # bare arm
        pygame.draw.circle(surface, (220, 30, 30), (w // 2 + 95, 22), 12) # red glove
        
        # Shorts (Dark gray)
        pygame.draw.rect(surface, (40, 40, 40), (w // 2 - 22, 105, 44, 25))
        
        # Legs
        pygame.draw.rect(surface, (240, 200, 160), (w // 2 - 18, 130, 10, 20))
        pygame.draw.rect(surface, (240, 200, 160), (w // 2 + 8, 130, 10, 20))
        
        # Boots
        pygame.draw.ellipse(surface, (10, 10, 10), (w // 2 - 23, 148, 18, 12))
        pygame.draw.ellipse(surface, (10, 10, 10), (w // 2 + 5, 148, 18, 12))
        
        return surface

    def play_sound(self, key):
        """
        Plays sound effect associated with key.
        """
        if key in self.sounds:
            try:
                self.sounds[key].play()
            except Exception as e:
                print(f"Error playing sound {key}: {e}")
