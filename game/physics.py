import numpy as np

class BallPhysics:
    """
    Simulates 3D football physics using quadratic Bezier curves
    and maps the coordinates to 2D Pygame screen space with perspective scaling.
    """
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # 2D Screen Reference Coordinates (Goal position and Penalty Spot)
        # These will be aligned with the stadium background image.
        self.penalty_spot = (screen_width // 2, int(screen_height * 0.82))
        
        self.goal_width = int(screen_width * 0.45)
        self.goal_height = int(screen_height * 0.35)
        self.goal_left = (screen_width - self.goal_width) // 2
        self.goal_right = self.goal_left + self.goal_width
        self.goal_top = int(screen_height * 0.25)
        self.goal_bottom = self.goal_top + self.goal_height
        
        self.goal_center_x = screen_width // 2
        
        # Ball sizing (shrinks as it recedes into the screen)
        self.start_radius = 35
        self.end_radius = 12

    def calculate_trajectory(self, direction, height, power, kick_angle=0.0):
        """
        Computes 3D trajectory parameters.
        direction: "LEFT", "CENTER", "RIGHT"
        height: "LOW", "MID", "HIGH"
        power: 0 - 100
        kick_angle: angle in degrees relative to vertical (determines curve)
        
        Returns a list of points (x, y, radius) in screen coordinates representing
        the trajectory from t=0 (kick) to t=1 (goal line).
        """
        # 3D Space Coordinates:
        # X: [-0.5, 0.5] (Goal line horizontal span: goal is between -0.5 and 0.5)
        # Y: [0.0, 0.5] (Goal line vertical height: ground is 0.0, crossbar is 0.4)
        # Z: [0.0, 1.0] (Z=0 is penalty spot, Z=1 is goal line)
        
        p_start = np.array([0.0, 0.0, 0.0])
        
        # Set target end point based on direction and height
        # Map direction to goal line X
        if direction == "LEFT":
            # Target left side of the net (but inside posts)
            x_target = np.random.uniform(-0.48, -0.22)
        elif direction == "RIGHT":
            # Target right side of the net
            x_target = np.random.uniform(0.22, 0.48)
        else:
            # Target center
            x_target = np.random.uniform(-0.15, 0.15)
            
        # Map height to goal line Y
        if height == "LOW":
            y_target = np.random.uniform(0.02, 0.08)
        elif height == "HIGH":
            # Near crossbar (0.4) or slightly over (miss)
            y_target = np.random.uniform(0.32, 0.45)
        else:
            # MID
            y_target = np.random.uniform(0.12, 0.28)
            
        p_end = np.array([x_target, y_target, 1.0])
        
        # Add curve effect based on kick_angle
        # Positive angle curves right, negative curves left
        curve_offset = np.clip(kick_angle / 90.0, -0.25, 0.25)
        
        # Add upward arc (gravity effect requires ball to go higher in the middle of flight)
        # High shots have higher apexes
        if height == "HIGH":
            arc_height = np.random.uniform(0.4, 0.6)
        elif height == "MID":
            arc_height = np.random.uniform(0.25, 0.35)
        else:
            arc_height = np.random.uniform(0.08, 0.15)
            
        # Control Point for Bezier curve (midpoint of depth Z=0.5)
        p_control = np.array([
            (p_start[0] + p_end[0]) / 2.0 + curve_offset,
            max(p_start[1], p_end[1]) + arc_height,
            0.5
        ])
        
        # Determine duration / number of frames based on power
        # High power (100) -> fast shot (approx 20 frames at 60 FPS = 0.33s)
        # Low power (15) -> slow shot (approx 60 frames at 60 FPS = 1.0s)
        num_frames = int(max(15, 60 - (power / 100.0) * 45))
        
        trajectory = []
        for i in range(num_frames + 1):
            t = i / num_frames
            
            # Bezier formula: P(t) = (1-t)^2 * P0 + 2*(1-t)*t * P1 + t^2 * P2
            p_t = (1 - t)**2 * p_start + 2 * (1 - t) * t * p_control + t**2 * p_end
            
            # Project 3D coordinate (x, y, z) to 2D screen coordinate
            x_screen, y_screen = self.project_to_screen(p_t[0], p_t[1], p_t[2])
            
            # Scale radius based on depth Z
            radius = int(self.start_radius * (1.0 - p_t[2]) + self.end_radius * p_t[2])
            
            trajectory.append({
                'x': x_screen,
                'y': y_screen,
                'z': p_t[2],
                'x_3d': p_t[0],
                'y_3d': p_t[1],
                'radius': radius
            })
            
        return trajectory

    def project_to_screen(self, x_3d, y_3d, z_3d):
        """
        Maps a 3D coordinate (x, y, z) to Pygame screen coordinates.
        - z_3d ranges from 0.0 (penalty spot) to 1.0 (goal plane).
        - x_3d maps to width (around goal_center_x at z=1).
        - y_3d maps to height (above ground, which is goal_bottom at z=1).
        """
        # Linear interpolation of screen center and limits based on depth (z)
        # At Z=0: we are at the penalty spot
        # At Z=1: we are in the goal plane
        
        # Screen position at Z=1 (Goal Plane projection)
        goal_x = self.goal_center_x + (x_3d * self.goal_width)
        goal_y = self.goal_bottom - (y_3d * self.goal_height)
        
        # Screen position at Z=0 (Start Plane projection)
        start_x = self.penalty_spot[0] + (x_3d * (self.screen_width * 0.15))
        start_y = self.penalty_spot[1]
        
        # Interpolate based on depth z_3d
        screen_x = int(start_x * (1.0 - z_3d) + goal_x * z_3d)
        screen_y = int(start_y * (1.0 - z_3d) + goal_y * z_3d)
        
        return screen_x, screen_y

    def is_inside_goal(self, x_3d, y_3d):
        """
        Checks if the final 3D ball coordinates enter the goal region.
        Goal posts are at X = [-0.44, 0.44], Crossbar is at Y = 0.38.
        """
        post_limit = 0.44
        bar_limit = 0.38
        
        # Check horizontal boundaries
        inside_posts = (-post_limit <= x_3d <= post_limit)
        # Check vertical boundary (must be above ground 0.0 and below crossbar 0.38)
        under_bar = (0.0 <= y_3d <= bar_limit)
        
        return inside_posts and under_bar
