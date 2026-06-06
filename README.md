# Vision Football Penalty Challenge

Vision Football Penalty Challenge is a premium, immersive desktop penalty shootout game built using Python, OpenCV, MediaPipe, NumPy, and Pygame. It uses real-time webcam tracking to put you in the shoes of either the striker (Penalty Kicker mode) or the goalie (Goalkeeper mode), simulating a classic Kinect-style gameplay experience on your computer.

---

## Features

### 1. Striker Mode (Penalty Kicker)
- **Webcam Kick Tracking**: Detects your leg swing velocity using MediaPipe Pose.
- **Directional Control**: Triggers shots to the Left, Center, or Right depending on the horizontal ankle displacement relative to your body center.
- **Shot Height**: Measures vertical ankle angle and height to launch low, mid-height, or soaring high shots.
- **Power Mapping**: Dynamically translates your leg swing speed to shot power (0–100%).
- **AI Goalkeeper**: Dives to save shots with difficulty-scaled reaction delays and save chances (Easy 30%, Medium 50%, Hard 70%).

### 2. Goalkeeper Mode
- **Dual Glove Tracking**: Normalizes your wrist positions via webcam body tracking and maps them to virtual goalie gloves.
- **Relative Reach Scaling**: Calibrates hand coordinates relative to your shoulders/hips center, letting you save corner shots with natural arm extensions.
- **Collision Detection**: If either glove overlaps the incoming ball trajectory near the goal line, a save is recorded.

### 3. Advanced Features
- **Shot Prediction AI**: A custom, lightweight K-Nearest Neighbors (KNN) classifier runs online, predicting where you're aiming (Left, Center, Right) based on body tilt, kick angle, and velocity before the ball reaches the net.
- **Skeletal Replay System**: After each shot, watch a slow-motion playback of the ball path and your neon skeleton lines.
- **Interactive Calibration Screen**: Directs you to stand in a T-pose for 3 seconds to measure shoulder width, leg length, and hip height, tuning velocity thresholds to fit your body size and camera distance.
- **Local Leaderboard**: Keeps track of high scores in `scores.json` and renders the top 10 on a leaderboard screen.
- **Hardware Fallbacks**: Automatically switches to mouse-controlled gameplay and procedurally synthesized 16-bit PCM sound wave buffers if a webcam or asset folder is missing, ensuring 100% playability on any machine out-of-the-box.

---

## Folder Structure

```text
football_game/
├── main.py                 # Game Orchestrator & State Machine
├── requirements.txt        # Python Dependencies
├── README.md               # Game Documentation
├── game/
│   ├── assets.py           # Asset loader & Audio synthesizer fallbacks
│   ├── menu.py             # Main Menu page
│   ├── penalty_mode.py     # Striker game mode logic
│   ├── goalkeeper_mode.py  # Goalkeeper game mode logic
│   ├── physics.py          # 3D quadratic Bezier curve physics
│   └── scoring.py          # High-score tracker and scoreboard manager
├── vision/
│   ├── pose_detector.py    # MediaPipe Pose tracker wrapper
│   ├── hand_detector.py    # MediaPipe Hands tracker wrapper
│   └── kick_detector.py    # Motion velocity and direction calculator
└── assets/                 # Image & Sound directories (Auto-fallback built-in)
    ├── ball.png
    ├── goalkeeper.png
    ├── goal.png
    ├── stadium_bg.jpg
    └── sounds/
```

---

## Installation & Setup

### Prerequisites
- Python 3.11+
- A working webcam connected to your computer.

### Step 1: Install Dependencies
Install all required libraries using pip:

```bash
pip install -r requirements.txt
```

### Step 2: Run the Game
Execute the entry-point script to start the game:

```bash
python main.py
```

---

## How to Play

1. **Setup Room Lighting**: Ensure your room is normally lit and that the camera has a clear view of your body.
2. **Launch Calibration**: In the Main Menu, select **CAMERA CALIBRATION**. Step back (approx. 6–8 feet) so your head down to your ankles is visible. Stand still in a **T-pose** (arms extended horizontally) for 3 seconds until the countdown completes.
3. **Striker Mode**:
   - Step back. Keep your hips and ankles visible.
   - Sweep your kicking leg forward swiftly to shoot.
   - Swing your leg outwards to shoot LEFT/RIGHT, or straight forward to shoot CENTER.
   - Lift your foot higher during the kick to shoot HIGH, or sweep it low for a LOW shot.
4. **Goalkeeper Mode**:
   - Put your arms out to block. Mapped goalie gloves will move as you move.
   - When the AI shoots, move your hands physically to intersect the ball trajectory at the goal line.
5. **Mouse Fallback Controls**: If no camera is available, use your mouse to play!
   - In *Striker Mode*, click anywhere in the lower pitch to kick (Left/Center/Right, Low/Mid/High).
   - In *Goalkeeper Mode*, the gloves track your mouse pointer.
