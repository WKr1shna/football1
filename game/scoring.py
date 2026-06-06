import json
import os
import time

class ScoreManager:
    """
    Manages current game score, penalties taken, goals scored, goalkeeper saves,
    and handles loading, updating, and saving the persistent local leaderboard.
    """
    def __init__(self, filename="scores.json"):
        # Get path of scores.json relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.filepath = os.path.join(current_dir, "..", filename)
        
        # Current game state
        self.attempts = 0
        self.max_attempts = 5
        self.goals = 0
        self.saves = 0
        
        # Leaderboard storage
        self.leaderboard = []
        self.load_leaderboard()

    def reset_score(self):
        """
        Resets current match variables.
        """
        self.attempts = 0
        self.goals = 0
        self.saves = 0

    def record_attempt(self, is_goal, is_save=False):
        """
        Updates counts after an attempt.
        """
        self.attempts += 1
        if is_goal:
            self.goals += 1
        if is_save:
            self.saves += 1

    def is_game_over(self):
        """
        Checks if the 5-penalty shootout is complete.
        """
        return self.attempts >= self.max_attempts

    def load_leaderboard(self):
        """
        Loads top scores from local JSON file.
        """
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r") as f:
                    self.leaderboard = json.load(f)
            except Exception as e:
                print(f"Error loading leaderboard: {e}")
                self.leaderboard = []
        else:
            self.leaderboard = []
            
        # Ensure it contains formatted list
        if not isinstance(self.leaderboard, list):
            self.leaderboard = []

    def save_leaderboard(self):
        """
        Saves leaderboard to local JSON file.
        """
        try:
            with open(self.filepath, "w") as f:
                json.dump(self.leaderboard, f, indent=4)
        except Exception as e:
            print(f"Error saving leaderboard: {e}")

    def add_score(self, player_name, mode, score, difficulty):
        """
        Adds a new score to the leaderboard, sorts, and limits to top 10.
        mode: "KICKER" or "GOALKEEPER"
        score: integer score (e.g. goals for striker, saves for goalkeeper)
        """
        entry = {
            "name": player_name,
            "mode": mode,
            "score": score,
            "difficulty": difficulty,
            "date": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.leaderboard.append(entry)
        
        # Sort scores: KICKER wants high goals, GOALKEEPER wants high saves.
        # Higher score is always better in either mode.
        self.leaderboard.sort(key=lambda x: x["score"], reverse=True)
        
        # Keep only Top 10
        self.leaderboard = self.leaderboard[:10]
        self.save_leaderboard()

    def get_top_scores(self, mode=None):
        """
        Returns top scores filtered by mode if specified.
        """
        if mode:
            return [x for x in self.leaderboard if x["mode"] == mode]
        return self.leaderboard
