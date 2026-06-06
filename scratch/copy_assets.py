import os
import shutil

# Source assets paths (from the Gemini session brain folder)
brain_dir = "/Users/kn/.gemini/antigravity/brain/f445069e-cfa7-48a4-9107-895319b3e3cd"

stadium_src = os.path.join(brain_dir, "stadium_bg_1780723848733.png")
ball_src = os.path.join(brain_dir, "ball_1780723866422.png")
goalkeeper_src = os.path.join(brain_dir, "goalkeeper_1780723884190.png")
goal_src = os.path.join(brain_dir, "goal_1780723901687.png")

# Target assets directory
target_dir = "/Users/kn/Desktop/mirai/SEM 2/Football/football_game/assets"
sounds_dir = os.path.join(target_dir, "sounds")

# Ensure target directories exist
os.makedirs(target_dir, exist_ok=True)
os.makedirs(sounds_dir, exist_ok=True)

# Copy and rename mapping
mappings = [
    (stadium_src, os.path.join(target_dir, "stadium_bg.jpg")), # We copy to jpg, Pygame handles it fine
    (ball_src, os.path.join(target_dir, "ball.png")),
    (goalkeeper_src, os.path.join(target_dir, "goalkeeper.png")),
    (goal_src, os.path.join(target_dir, "goal.png"))
]

for src, dst in mappings:
    if os.path.exists(src):
        try:
            shutil.copy2(src, dst)
            print(f"Copied {os.path.basename(src)} -> {os.path.basename(dst)}")
        except Exception as e:
            print(f"Error copying {src}: {e}")
    else:
        print(f"Source file not found: {src}")

print("Asset copying complete!")
