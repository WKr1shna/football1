// Lightweight K-Nearest Neighbors Classifier for Shot Prediction
class ShotPredictionAI {
    constructor() {
        this.k = 3;
        // Prepopulate training data: [kick_angle, ankle_velocity, body_orientation]
        // Labels: 0 = LEFT, 1 = CENTER, 2 = RIGHT
        this.trainData = [
            { f: [15.0, 0.15, -5.0], l: 0 },
            { f: [25.0, 0.20, -10.0], l: 0 },
            { f: [0.0, 0.12, 0.0], l: 1 },
            { f: [2.0, 0.18, 1.0], l: 1 },
            { f: [-15.0, 0.15, 5.0], l: 2 },
            { f: [-25.0, 0.20, 10.0], l: 2 }
        ];
    }
    
    trainSample(kickAngle, ankleVelocity, bodyOrientation, label) {
        this.trainData.push({
            f: [parseFloat(kickAngle), parseFloat(ankleVelocity), parseFloat(bodyOrientation)],
            l: parseInt(label)
        });
        if (this.trainData.length > 100) {
            this.trainData.shift();
        }
    }
    
    predict(kickAngle, ankleVelocity, bodyOrientation) {
        if (this.trainData.length === 0) return { label: 1, confidence: 1.0 };
        
        const x = [kickAngle, ankleVelocity, bodyOrientation];
        const numFeatures = 3;
        
        // Calculate feature means & stds dynamically
        const means = [0, 0, 0];
        const stds = [0, 0, 0];
        
        for (let j = 0; j < numFeatures; j++) {
            let sum = 0;
            for (let i = 0; i < this.trainData.length; i++) {
                sum += this.trainData[i].f[j];
            }
            means[j] = sum / this.trainData.length;
            
            let varSum = 0;
            for (let i = 0; i < this.trainData.length; i++) {
                varSum += Math.pow(this.trainData[i].f[j] - means[j], 2);
            }
            stds[j] = Math.sqrt(varSum / this.trainData.length) || 1e-5;
        }
        
        // Z-score normalize target and training samples
        const xNorm = x.map((val, j) => (val - means[j]) / stds[j]);
        
        const distances = this.trainData.map(sample => {
            const sampleNorm = sample.f.map((val, j) => (val - means[j]) / stds[j]);
            const dist = Math.sqrt(sampleNorm.reduce((acc, val, j) => acc + Math.pow(val - xNorm[j], 2), 0));
            return { dist, label: sample.l };
        });
        
        // Sort by distance ascending
        distances.sort((a, b) => a.dist - b.dist);
        
        // Get k nearest neighbors and vote
        const kNeighbors = distances.slice(0, this.k);
        const votes = [0, 0, 0];
        kNeighbors.forEach(n => votes[n.label]++);
        
        const maxVoteLabel = votes.indexOf(Math.max(...votes));
        const confidence = votes[maxVoteLabel] / this.k;
        
        return { label: maxVoteLabel, confidence };
    }
}

// AssetManager class to validate dimensions and transparency of visual assets
class AssetManager {
    constructor() {
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
    }

    validateImage(name, img) {
        if (!img) return Promise.resolve(false);
        return new Promise((resolve) => {
            const check = () => {
                if (img.naturalWidth === 0 || img.naturalHeight === 0) {
                    console.warn(`[AssetManager] Image "${name}" has invalid dimensions: ${img.naturalWidth}x${img.naturalHeight}`);
                    resolve(false);
                    return;
                }

                // Check alpha transparency
                this.canvas.width = img.naturalWidth;
                this.canvas.height = img.naturalHeight;
                this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
                this.ctx.drawImage(img, 0, 0);

                try {
                    const imgData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height).data;
                    let hasAlpha = false;
                    // Check every fourth pixel's alpha value
                    for (let i = 3; i < imgData.length; i += 16) {
                        if (imgData[i] < 255) {
                            hasAlpha = true;
                            break;
                        }
                    }
                    if (name !== 'stadium' && !hasAlpha) {
                        console.warn(`[AssetManager] Warning: Image "${name}" is expected to have transparency but none was detected.`);
                    } else {
                        console.log(`[AssetManager] Loaded and validated "${name}" (${img.naturalWidth}x${img.naturalHeight}, hasAlpha: ${hasAlpha})`);
                    }
                    resolve(true);
                } catch (e) {
                    // CORS issues under file:// or other browser context policies
                    console.warn(`[AssetManager] Could not read image data for "${name}" (CORS issue or context error):`, e.message);
                    resolve(true); // Treat as valid to avoid blocking the game
                }
            };

            if (img.complete) {
                check();
            } else {
                img.onload = check;
                img.onerror = () => {
                    console.warn(`[AssetManager] Failed to load image "${name}" at source: ${img.src}`);
                    resolve(false);
                };
            }
        });
    }
}

// Main Game Orchestrator class
class GameOrchestrator {
    constructor() {
        // Subsystems
        this.audio = new GameAudioManager();
        this.physics = new BallPhysics(1024, 768);
        this.vision = new GameVisionManager();
        this.predictor = new ShotPredictionAI();
        
        // Canvas setups
        this.canvas = document.getElementById('game-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.calibCanvas = document.getElementById('calibration-canvas');
        this.calibCtx = this.calibCanvas.getContext('2d');
        this.menuWebcamCanvas = document.getElementById('menu-webcam-canvas');
        this.menuWebcamCtx = this.menuWebcamCanvas.getContext('2d');
        this.gameWebcamCanvas = document.getElementById('game-webcam-canvas');
        this.gameWebcamCtx = this.gameWebcamCanvas.getContext('2d');
        
        // Offscreen canvas for procedural caching
        this.stadiumCanvas = document.createElement('canvas');
        this.stadiumCanvas.width = 1024;
        this.stadiumCanvas.height = 768;
        this.stadiumCtx = this.stadiumCanvas.getContext('2d');
        this.renderProceduralStadium(this.stadiumCtx, 1024, 768);
        
        // Images preloading
        this.images = {
            stadium: new Image(),
            ball: new Image(),
            goalkeeper: new Image(), // kept as fallback
            goalkeeperIdle: new Image(),
            goalkeeperDiveLeft: new Image(),
            goalkeeperDiveRight: new Image(),
            goalkeeperSave: new Image(),
            goal: new Image()
        };
        this.images.stadium.src = 'assets/stadium.png';
        this.images.ball.src = 'assets/football.png';
        this.images.goalkeeper.src = 'assets/goalkeeper_idle.png';
        this.images.goalkeeperIdle.src = 'assets/goalkeeper_idle.png';
        this.images.goalkeeperDiveLeft.src = 'assets/goalkeeper_dive_left.png';
        this.images.goalkeeperDiveRight.src = 'assets/goalkeeper_dive_right.png';
        this.images.goalkeeperSave.src = 'assets/goalkeeper_save.png';
        this.images.goal.src = 'assets/goal.png';
        
        // Validate assets
        this.assetManager = new AssetManager();
        Object.entries(this.images).forEach(([name, img]) => {
            this.assetManager.validateImage(name, img);
        });
        
        // Game States
        // "MENU", "CALIBRATION", "PLAY_STRIKER", "PLAY_GOALKEEPER", "NAME_INPUT", "LEADERBOARD"
        this.state = "MENU";
        this.menuIndex = 0;
        this.menuOptions = ["STRIKER", "GOALKEEPER", "DIFFICULTY", "CALIBRATION", "LEADERBOARD"];
        this.difficultyOptions = ["EASY", "MEDIUM", "HARD"];
        this.difficultyIndex = 1; // MEDIUM default
        
        // Calibration Data
        this.calibrationData = {
            shoulderWidth: 0.15,
            armLength: 0.45,
            bodyCenter: 0.5,
            hipHeight: 0.60,
            // GK neutral mapping (set during inline GK calibration)
            gkNeutralX: 0.5,   // normalized hand center x at natural stance
            gkNeutralY: 0.45,  // normalized hand center y at natural stance
            gkReachTop: 0.2,   // highest wrist y reached (raised hands)
            gkReachBottom: 0.7, // lowest wrist y reached (arms low)
            // Striker calibration parameters
            strikerNeutralHipX: 0.5,
            strikerNeutralHipY: 0.65,
            legLength: 0.35
        };
        this.calibrated = false;
        this.gkCalibrated = false;  // tracks if GK inline calibration was done
        this.strikerCalibrated = false;
        
        // Calibration active parameters
        this.calibStartTime = 0;
        this.calibSamples = [];
        this.gkNeutralSamples = [];  // used by inline GK calibration
        this.strikerCalibSamples = [];
        this.strikerCalibPhaseTimer = 0;
        
        // Goalkeeper Mode Hand Tracking History & Dive Detection
        this.gkHandHistory = [];
        this.handVelocityX = 0;
        this.gkDiveState = null;
        this.gkDiveTimer = 0;
        this.lastLeftVel = 0;
        this.lastRightVel = 0;
        this.debugMode = false;
        this.trainingMode = false; // T key toggles training overlay
        this.kickTime = 0;
        
        // ── Biomechanical Kick State Machine ────────────────────────────
        // Phases: IDLE → BACKSWING → STRIKE → FOLLOW_THROUGH
        this.kickPhase = "IDLE";
        this.kickPhaseTimer = 0;
        this.kickWindUpTime = 0;
        this.currentAimZone = "MID_CENTER";
        this.strikerFlashTimer = 0;
        
        // Per-frame biomech tracking (hip, knee, ankle per leg)
        this.bioHistory = {
            left:  [],  // { time, hipX, hipY, kneeX, kneeY, ankleX, ankleY }
            right: []
        };
        this.bioHistoryLen = 8;
        this.dominantLeg = null;     // auto-detected: 'left' or 'right'
        this.dominantLegVotes = { left: 0, right: 0 };
        this.backswingAnchor = null; // { ankleX, ankleY, hipX, hipY } at start of backswing
        this.strikeSnapshot = null;  // biomech snapshot at moment of strike
        
        // Physical goalkeeper colliders (set by determineAiGoalkeeperDive)
        // Each collider: { x, y, w, h } in screen-space
        this.keeperColliders = [];
        this.keeperDiveHeight = "MID";
        
        // Goalkeeper AI & animation state variables
        this.keeperAiState = "READING"; // READING, ANTICIPATING, DIVING, LANDED
        this.keeperPredictedTarget = "MID_CENTER";
        this.keeperPredictionConfidence = 50;
        this.keeperCommitTime = 0;
        this.keeperReactionStarted = false;
        this.lastPerceptionUpdateTime = 0;
        this._lastBodyLean = 0;
        
        // Purely Visual Animation State Machine
        this.keeperAnim = {
            state: 0, // 0: IDLE, 1: ANTICIPATE, 2: PUSH, 3: DIVE, 4: SAVE, 5: LAND, 6: RECOVER
            dir: "CENTER",
            height: "MID",
            startTime: 0,
            elapsed: 0,
            visX: 512,
            visY: 500,
            headRot: 0,
            torsoRot: 0,
            leftArmRot: 0,
            rightArmRot: 0,
            leftLegRot: 0,
            rightLegRot: 0,
            shadowScale: 1.0,
            shadowWidth: 45
        };
        
        // Gameplay Variables
        this.attempts = 0;
        this.maxAttempts = 5;
        this.goals = 0;
        this.saves = 0;
        this.gameModeType = "STRIKER"; // "STRIKER" or "GOALKEEPER"
        
        // Active attempt details
        this.gameState = "WAITING_KICK"; // WAITING_KICK / WAITING_SHOT / BALL_FLIGHT / RESULT_CELEBRATION / COMPLETED
        this.gameStateTimer = 0;
        this.ballScreenPos = { x: this.physics.penaltySpot.x, y: this.physics.penaltySpot.y };
        this.ballRadius = this.physics.startRadius;
        this.ballPos3d = [0, 0, 0];
        
        this.ballTrajectory = [];
        this.trajectoryIndex = 0;
        
        // Goalkeeper details
        this.keeperX = 512;
        this.keeperY = this.physics.goalBottom - 20;
        this.keeperTargetX = 512;
        this.keeperTargetY = this.physics.goalBottom - 20;
        this.keeperDiveDir = "CENTER";
        this.keeperRotation = 0;
        this.aiDiveTriggerIndex = 0;
        
        // Goalkeeper glove details (GK Mode)
        this.leftGlovePos = { x: this.physics.goalLeft + 50, y: this.physics.goalBottom - 100 };
        this.rightGlovePos = { x: this.physics.goalRight - 50, y: this.physics.goalBottom - 100 };
        this.gloveRadius = 45;
        
        // AI Shot settings (GK Mode)
        this.aiShotDirection = "CENTER";
        this.aiShotHeight = "MID";
        this.aiShotPower = 60;
        this.aiShotAngle = 0.0;
        
        // Outcome parameters
        this.kickInfo = null;
        this.outcomeText = "";
        this.outcomeColor = "#ffffff";
        this.hasPlayedOutcomeSound = false;
        
        // Motion / Kick detection parameters
        this.ankleHistory = {
            left: [],
            right: []
        };
        this.historyLen = 5;
        this.speedThreshold = 0.04;
        this.coolDownTime = 1500; // ms
        this.lastKickTime = 0;
        // Local name input variables
        this.playerName = "";
        this.qualifyingScore = 0;
        this.leaderboardMode = "STRIKER";
        
        // Fallbacks
        this.mousePos = { x: 512, y: 384 };
        this.isMouseDown = false;
        
        // Visual effects state
        this.cameraShakeTimer = 0;
        this.screenFlashTimer = 0;
        this.ballHistory = [];
        this.ballSpinAngle = 0;
        
        // FPS meter
        this.lastFpsUpdate = 0;
        this.frameCount = 0;
        this.fps = 60;
        
        this.bindEvents();
    }

    async start() {
        // Init Web Audio
        this.audio.init();
        
        // Initialize MediaPipe Pose (asynchronous, non-blocking)
        this.vision.init((joints, results) => {
            this.processVisionFrame(joints, results);
        });
        
        // Start Loop
        requestAnimationFrame((t) => this.loop(t));
    }

    bindEvents() {
        // Keyboard navigation
        window.addEventListener('keydown', (e) => {
            this.handleKeyDown(e);
        });

        // Mouse click navigation & mouse control fallbacks
        window.addEventListener('mousemove', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            // Scale coordinates appropriately to match 1024x768 canvas space
            this.mousePos.x = (e.clientX - rect.left) * (this.canvas.width / rect.width);
            this.mousePos.y = (e.clientY - rect.top) * (this.canvas.height / rect.height);
        });

        window.addEventListener('mousedown', (e) => {
            this.isMouseDown = true;
            this.handleMouseDown(e);
        });
        
        window.addEventListener('mouseup', () => {
            this.isMouseDown = false;
        });

        // Menu button hover events
        const buttons = document.querySelectorAll('.menu-btn');
        buttons.forEach(btn => {
            btn.addEventListener('mouseenter', () => {
                const idx = parseInt(btn.dataset.index);
                if (this.state === "MENU") {
                    this.menuIndex = idx;
                    this.updateMenuButtons();
                    this.audio.playSound('kick'); // subtle click
                }
            });
            btn.addEventListener('click', () => {
                if (this.state === "MENU") {
                    this.executeSelection();
                }
            });
        });

        // Register Name Input events
        const nameInput = document.getElementById('player-name-input');
        nameInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                const name = nameInput.value.trim().toUpperCase() || "PLAYER";
                this.addLeaderboardEntry(name);
                nameInput.value = "";
                this.showView("leaderboard-view");
                this.state = "LEADERBOARD";
                this.audio.playSound('goal');
            }
        });
    }

    showView(id) {
        document.querySelectorAll('.state-view').forEach(view => {
            view.classList.remove('active');
        });
        document.getElementById(id).classList.add('active');
    }

    loop(timestamp) {
        this.frameCount++;
        if (timestamp - this.lastFpsUpdate >= 1000) {
            this.fps = Math.round((this.frameCount * 1000) / (timestamp - this.lastFpsUpdate));
            this.frameCount = 0;
            this.lastFpsUpdate = timestamp;
            this.updateFpsOverlay();
        }

        // Draw views based on active state
        if (this.state === "MENU") {
            this.runMenuLoop();
        } else if (this.state === "CALIBRATION") {
            this.runCalibrationLoop();
        } else if (this.state === "PLAY_STRIKER") {
            this.runStrikerLoop();
        } else if (this.state === "PLAY_GOALKEEPER") {
            this.runKeeperLoop();
        } else if (this.state === "LEADERBOARD") {
            this.runLeaderboardLoop();
        } else if (this.state === "NAME_INPUT") {
            this.runNameInputLoop();
        }
        
        // Screen flash overlay (on goal conceded)
        if (this.screenFlashTimer && Date.now() - this.screenFlashTimer < 350) {
            const ctx = this.ctx;
            const elapsed = Date.now() - this.screenFlashTimer;
            const alpha = Math.max(0, 0.6 - elapsed / 350 * 0.6);
            ctx.save();
            ctx.fillStyle = `rgba(255, 50, 50, ${alpha})`;
            ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
            ctx.restore();
        }

        requestAnimationFrame((t) => this.loop(t));
    }

    updateFpsOverlay() {
        const cam = this.vision.webcamConnected ? "ON" : "MOCK (MOUSE)";
        const cal = this.calibrated ? "ACTIVE" : "DEFAULT";
        document.getElementById('fps-counter').innerText = 
            `FPS: ${this.fps} | WEBCAM: ${cam} | CALIBRATION: ${cal}`;
    }

    // STATE LOOP: MENU
    runMenuLoop() {
        // Render webcam feed mirrored inside the preview box
        const w = this.menuWebcamCanvas.width;
        const h = this.menuWebcamCanvas.height;
        this.menuWebcamCtx.clearRect(0, 0, w, h);
        this.vision.drawSkeleton(this.menuWebcamCtx, w, h, '#64ff64', true);
    }

    updateMenuButtons() {
        const buttons = document.querySelectorAll('.menu-btn');
        buttons.forEach((btn, idx) => {
            if (idx === this.menuIndex) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });
    }

    handleKeyDown(e) {
        if (e.key.toLowerCase() === 'd') {
            this.debugMode = !this.debugMode;
            this.audio.playSound('kick');
        }
        if (e.key.toLowerCase() === 't') {
            this.trainingMode = !this.trainingMode;
            this.audio.playSound('kick');
        }
        
        if (this.state === "MENU") {
            if (e.key === "ArrowUp") {
                this.menuIndex = (this.menuIndex - 1 + this.menuOptions.length) % this.menuOptions.length;
                this.updateMenuButtons();
                this.audio.playSound('kick');
            } else if (e.key === "ArrowDown") {
                this.menuIndex = (this.menuIndex + 1) % this.menuOptions.length;
                this.updateMenuButtons();
                this.audio.playSound('kick');
            } else if (e.key === "Enter") {
                this.executeSelection();
            }
        } else if (this.state === "CALIBRATION") {
            if (e.key === "Escape") {
                this.showView("menu-view");
                this.state = "MENU";
                this.audio.playSound('kick');
            }
        } else if (this.state === "PLAY_STRIKER" || this.state === "PLAY_GOALKEEPER") {
            if (e.key === "Escape") {
                this.showView("menu-view");
                this.state = "MENU";
                this.audio.playSound('kick');
            }
        } else if (this.state === "LEADERBOARD") {
            // press any key to go back
            this.showView("menu-view");
            this.state = "MENU";
            this.audio.playSound('kick');
        }
    }

    handleMouseDown(e) {
        if (this.state === "PLAY_STRIKER" && !this.vision.webcamConnected && this.gameState === "WAITING_KICK") {
            const mx = this.mousePos.x;
            const my = this.mousePos.y;
            
            const gl = this.physics.strikerGoalLeft;
            const gw = this.physics.strikerGoalWidth;
            const gt = this.physics.strikerGoalTop;
            const gh = this.physics.strikerGoalHeight;
            
            // Map horizontal click coordinates to LEFT, CENTER, RIGHT
            let dir = "CENTER";
            if (mx < gl + gw / 3) dir = "LEFT";
            else if (mx > gl + (2 * gw) / 3) dir = "RIGHT";
            
            // Map vertical click coordinates to TOP, MID, BOTTOM
            let height = "MID";
            if (my < gt + gh / 3) height = "TOP";
            else if (my > gt + (2 * gh) / 3) height = "BOTTOM";
            
            const targetZone = `${height}_${dir}`;
            
            const power = 75 + Math.floor(Math.random() * 20); // 75-95%
            const angle = -10 + Math.random() * 20;
            
            this.lastKickTime = Date.now();
            this.kickTime = Date.now();
            this.audio.playSound('kick');
            
            this.kickInfo = {
                leg: 'right',
                velocity: 0.08,
                power: power,
                direction: dir,
                height: height,
                kick_angle: angle,
                body_orientation: 0.0,
                prediction: dir,
                confidence: 1.0
            };
            
            this.ballTrajectory = this.physics.calculateTrajectory(targetZone, power, angle);
            this.trajectoryIndex = 0;
            this.determineAiGoalkeeperDive(targetZone);
            
            // Update power meter to simulate peak power
            this.updatePowerBar(power);
            
            const promptEl = document.getElementById('game-prompt');
            if (promptEl) promptEl.classList.remove('active');
            
            this.gameState = "BALL_FLIGHT";
        }
    }

    executeSelection() {
        this.audio.playSound('save');
        const selection = this.menuOptions[this.menuIndex];
        
        if (selection === "STRIKER") {
            this.resetGame("STRIKER");
        } else if (selection === "GOALKEEPER") {
            this.resetGame("GOALKEEPER");
        } else if (selection === "DIFFICULTY") {
            this.difficultyIndex = (this.difficultyIndex + 1) % this.difficultyOptions.length;
            const diff = this.difficultyOptions[this.difficultyIndex];
            document.getElementById('btn-difficulty').innerText = `DIFFICULTY: ${diff}`;
            this.audio.playSound('save');
        } else if (selection === "CALIBRATION") {
            this.showView("calibration-view");
            this.state = "CALIBRATION";
            this.calibStartTime = 0;
            this.calibSamples = [];
        } else if (selection === "LEADERBOARD") {
            this.renderLeaderboardTable();
            this.showView("leaderboard-view");
            this.state = "LEADERBOARD";
        }
    }

    resetGame(mode) {
        this.gameModeType = mode;
        this.attempts = 0;
        this.goals = 0;
        this.saves = 0;
        
        const view = document.getElementById('game-view');
        const diffTag = document.getElementById('game-difficulty-tag');
        const roundText = document.getElementById('hud-round-text');
        const stat1Label = document.getElementById('hud-stat1-label');
        const stat2Label = document.getElementById('hud-stat2-label');
        
        if (mode === "STRIKER") {
            if (view) view.className = "state-view striker-mode";
            if (diffTag) diffTag.innerText = `AI: ${this.difficultyOptions[this.difficultyIndex]}`;
            if (roundText) roundText.innerText = `ROUND 1 / ${this.maxAttempts}`;
            if (stat1Label) stat1Label.innerText = 'GOALS:';
            if (stat2Label) stat2Label.innerText = 'MISSES:';
            this.resetAttempt();
            this.showView("game-view");
            this.state = "PLAY_STRIKER";
            this.gameState = "STRIKER_CALIBRATION";
            this.strikerCalibPhaseTimer = Date.now();
            this.strikerCalibSamples = [];
        } else {
            if (view) view.className = "state-view keeper-mode";
            if (diffTag) diffTag.innerText = `DIFF: ${this.difficultyOptions[this.difficultyIndex]}`;
            if (roundText) roundText.innerText = `ROUND 1 / ${this.maxAttempts}`;
            if (stat1Label) stat1Label.innerText = 'SAVES:';
            if (stat2Label) stat2Label.innerText = 'GOALS:';
            this.resetAttempt();
            // Override: begin with GK inline calibration
            this.gameState = "GK_CALIBRATION";
            this.gkCalibPhaseTimer = Date.now();
            this.gkNeutralSamples = [];
            this.showView("game-view");
            this.state = "PLAY_GOALKEEPER";
        }
    }

    resetAttempt() {
        this.ballScreenPos = { x: this.physics.penaltySpot.x, y: this.physics.penaltySpot.y };
        this.ballRadius = this.physics.startRadius;
        this.ballPos3d = [0, 0, 0];
        
        this.ballTrajectory = [];
        this.trajectoryIndex = 0;
        this.ballHistory = [];
        this.ballSpinAngle = 0;
        
        // Goalkeeper base position
        this.keeperX = 512;
        if (this.gameModeType === "STRIKER") {
            // In first person striker view, goalkeeper starts at striker goal line
            this.keeperY = this.physics.strikerGoalBottom;
            this.keeperTargetY = this.physics.strikerGoalBottom;
        } else {
            this.keeperY = this.physics.gkGoalBottom - 20;
            this.keeperTargetY = this.physics.gkGoalBottom - 20;
        }
        this.keeperTargetX = 512;
        this.keeperDiveDir = "CENTER";
        this.keeperRotation = 0;
        
        // Reset goalkeeper AI & animation state
        this.keeperAiState = "READING";
        this.keeperPredictedTarget = "";
        this.keeperPredictionConfidence = 0;
        this.keeperCommitTime = 0;
        this.keeperReactionStarted = false;
        this.lastPerceptionUpdateTime = 0;
        this._lastBodyLean = 0;
        
        // Reset striker variables
        this.kickPhase = "IDLE";
        this.kickPhaseTimer = 0;
        this.currentAimZone = "MID_CENTER";
        this.strikerFlashTimer = 0;
        this.backswingAnchor = null;
        this.strikeSnapshot = null;
        this._biomechSnapshot = null;
        this._keeperColliderDef = [];
        this.keeperColliders = [];
        this.bioHistory = { left: [], right: [] };
        this.dominantLegVotes = { left: 0, right: 0 };
        
        this.aiDiveTriggerIndex = 0;
        
        this.kickInfo = null;
        this.outcomeText = "";
        this.hasPlayedOutcomeSound = false;
        
        // Hide announcements & countdowns
        document.getElementById('game-prompt').classList.remove('active');
        document.getElementById('game-outcome').classList.remove('active');
        
        // Reset power bar
        this.updatePowerBar(0);
        
        // Reset goalkeeper dive states
        this.gkDiveState = null;
        this.gkDiveTimer = 0;
        this.gkHandHistory = [];
        
        // Reset visual effects
        this.cameraShakeTimer = 0;
        this.screenFlashTimer = 0;
        this.ballHistory = [];
        this.ballSpinAngle = 0;
        
        if (this.gameModeType === "STRIKER") {
            this.gameState = "WAITING_KICK_COUNTDOWN";
            this.countdownValue = 3;
            this.countdownTimer = Date.now();
            
            const cdEl = document.getElementById('game-countdown');
            if (cdEl) {
                cdEl.innerText = "3";
                cdEl.style.display = 'block';
                cdEl.classList.add('active');
            }
            
            const promptEl = document.getElementById('game-prompt');
            if (promptEl) {
                promptEl.innerText = "PREPARE YOUR SHOT";
                promptEl.classList.add('active');
            }
        } else {
            this.gameState = "WAITING_SHOT";
            this.gameStateTimer = Date.now();
            
            const cdEl = document.getElementById('game-countdown');
            if (cdEl) {
                cdEl.style.display = 'none';
                cdEl.classList.remove('active');
            }
            
            const promptEl = document.getElementById('game-prompt');
            if (promptEl) {
                promptEl.innerText = "AI PREPARING SHOT... GET READY!";
                promptEl.classList.add('active');
            }
        }
        
        // Update the new top-bar HUD stat values
        const stat1Val = document.getElementById('hud-stat1-val');
        const stat2Val = document.getElementById('hud-stat2-val');
        const roundText = document.getElementById('hud-round-text');
        
        if (this.gameModeType === "STRIKER") {
            if (stat1Val) stat1Val.innerText = this.goals;
            if (stat2Val) stat2Val.innerText = this.attempts - this.goals;
        } else {
            if (stat1Val) stat1Val.innerText = this.saves;
            if (stat2Val) stat2Val.innerText = this.attempts - this.saves;
        }
        if (roundText) roundText.innerText = `ROUND ${this.attempts + 1} / ${this.maxAttempts}`;
    }

    // STATE LOOP: CAMERA CALIBRATION
    runCalibrationLoop() {
        const ctx = this.calibCtx;
        const w = this.calibCanvas.width;
        const h = this.calibCanvas.height;
        ctx.clearRect(0, 0, w, h);
        
        // Draw mirrored webcam stream with standard lines overlay
        this.vision.drawSkeleton(ctx, w, h, '#ffe650', true);
        
        // Draw stencils to guide user
        ctx.lineWidth = 3;
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
        ctx.beginPath();
        ctx.arc(w / 2, h * 0.22, 45, 0, 2 * Math.PI); // head circle
        ctx.moveTo(w * 0.3, h * 0.25);
        ctx.lineTo(w * 0.3, h * 0.45); // Left raised arm outline
        ctx.moveTo(w * 0.7, h * 0.25);
        ctx.lineTo(w * 0.7, h * 0.45); // Right raised arm outline
        ctx.moveTo(w / 2, h * 0.22 + 45);
        ctx.lineTo(w / 2, h * 0.75); // spine
        ctx.stroke();
        
        // Validate pose skeleton
        const results = this.vision.poseResults;
        const essentialKeys = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip', 'left_ankle', 'right_ankle'];
        const joints = this.vision.getKeyJoints(results);
        
        // Essential joints visible and hands raised above shoulders
        const skeletonFullyVisible = results && results.poseLandmarks && 
            essentialKeys.every(k => joints[k] && joints[k].visibility > 0.60);
            
        const armsRaised = skeletonFullyVisible && joints['left_wrist'] && joints['right_wrist'] &&
            joints['left_wrist'].y_norm < joints['left_shoulder'].y_norm &&
            joints['right_wrist'].y_norm < joints['right_shoulder'].y_norm;
            
        const titleEl = document.getElementById('calibration-title');
        const subtitleEl = document.getElementById('calibration-subtitle');
        
        if (armsRaised) {
            if (this.calibStartTime === 0) {
                this.calibStartTime = Date.now();
                this.audio.playSound('kick');
            }
            
            const elapsed = (Date.now() - this.calibStartTime) / 1000.0;
            const countdown = 3.0 - elapsed;
            
            this.calibSamples.push(joints);
            
            if (countdown <= 0) {
                this.completeCalibration();
                this.showView("menu-view");
                this.state = "MENU";
                return;
            }
            
            titleEl.innerText = "HOLD THE POSITION STILL...";
            titleEl.className = "glow-yellow";
            subtitleEl.innerText = `Calibrating body proportions in ${Math.max(1, Math.ceil(countdown))}...`;
            
            // Draw visual loading bar
            ctx.fillStyle = '#ffe650';
            ctx.fillRect(50, h - 30, (w - 100) * (elapsed / 3.0), 10);
        } else {
            // Reset calibration
            this.calibStartTime = 0;
            this.calibSamples = [];
            
            titleEl.innerText = "STAND IN THE CENTER AND RAISE BOTH ARMS";
            titleEl.className = "";
            subtitleEl.innerText = "Ensure head, shoulders, hips, knees, ankles and raised wrists are visible.";
        }
    }

    completeCalibration() {
        if (this.calibSamples.length === 0) return;
        
        const shoulders = [];
        const armLengths = [];
        const bodyCenters = [];
        const hips = [];
        
        this.calibSamples.forEach(joints => {
            const ls = joints['left_shoulder'];
            const rs = joints['right_shoulder'];
            shoulders.push(Math.abs(ls.x_norm - rs.x_norm));
            
            const lh = joints['left_hip'];
            const rh = joints['right_hip'];
            bodyCenters.push((lh.x_norm + rh.x_norm) / 2.0);
            hips.push(lh.y_norm);
            
            // Arm length: shoulder to elbow + elbow to wrist
            const le = joints['left_elbow'];
            const lw = joints['left_wrist'];
            const re = joints['right_elbow'];
            const rw = joints['right_wrist'];
            
            if (le && lw && re && rw) {
                const leftArm = Math.sqrt(Math.pow(le.x_norm - ls.x_norm, 2) + Math.pow(le.y_norm - ls.y_norm, 2)) +
                               Math.sqrt(Math.pow(lw.x_norm - le.x_norm, 2) + Math.pow(lw.y_norm - le.y_norm, 2));
                const rightArm = Math.sqrt(Math.pow(re.x_norm - rs.x_norm, 2) + Math.pow(re.y_norm - rs.y_norm, 2)) +
                                Math.sqrt(Math.pow(rw.x_norm - re.x_norm, 2) + Math.pow(rw.y_norm - re.y_norm, 2));
                armLengths.push((leftArm + rightArm) / 2.0);
            }
        });
        
        const avg = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
        
        this.calibrationData = {
            shoulderWidth: avg(shoulders),
            armLength: armLengths.length > 0 ? avg(armLengths) : 0.45,
            bodyCenter: avg(bodyCenters),
            hipHeight: avg(hips)
        };
        this.calibrated = true;
        this.audio.playSound('goal');
        console.log("Calibration successful:", this.calibrationData);
        
        // Adjust speed threshold relative to calibrated arm/leg reach
        this.speedThreshold = 0.04 * (this.calibrationData.armLength / 0.45);
        this.speedThreshold = Math.max(0.02, Math.min(0.08, this.speedThreshold));
    }

    completeGkCalibration() {
        if (this.gkNeutralSamples.length === 0) {
            // No webcam samples — use sensible screen-space defaults
            this.calibrationData.gkNeutralX = 0.5;
            this.calibrationData.gkNeutralY = 0.45;
            this.calibrationData.gkReachTop = 0.2;
            this.calibrationData.gkReachBottom = 0.72;
            this.gkCalibrated = false;
            return;
        }
        
        const avg = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
        const xs = this.gkNeutralSamples.map(s => s.x);
        const ys = this.gkNeutralSamples.map(s => s.y);
        
        this.calibrationData.gkNeutralX = avg(xs);
        this.calibrationData.gkNeutralY = avg(ys);
        
        // Reach: estimate from pose — top is shoulder line, bottom is hip level
        // Use collected wrist y values; clip to ±25% of neutral for safety
        const ny = this.calibrationData.gkNeutralY;
        this.calibrationData.gkReachTop    = Math.max(0.05, ny - 0.28);  // ~28% above neutral
        this.calibrationData.gkReachBottom = Math.min(0.95, ny + 0.28);  // ~28% below neutral
        
        this.gkCalibrated = true;
        this.gkNeutralSamples = [];
        console.log("GK Calibration complete:", this.calibrationData);
    }

    // PROCESS KEY CV UPDATES
    processVisionFrame(joints, results) {
        if (this.state === "PLAY_STRIKER") {
            if (this.gameState === "STRIKER_CALIBRATION") {
                if (joints['left_hip'] && joints['right_hip'] && joints['left_ankle'] && joints['right_ankle']) {
                    const lh = joints['left_hip'];
                    const rh = joints['right_hip'];
                    const la = joints['left_ankle'];
                    const ra = joints['right_ankle'];
                    
                    this.strikerCalibSamples.push({
                        hipX: (lh.x_norm + rh.x_norm) / 2.0,
                        hipY: (lh.y_norm + rh.y_norm) / 2.0,
                        leftLegLen: Math.abs(lh.y_norm - la.y_norm),
                        rightLegLen: Math.abs(rh.y_norm - ra.y_norm)
                    });
                    
                    if (Date.now() - this.strikerCalibPhaseTimer > 3000) {
                        this.completeStrikerCalibration();
                        this.gameState = "WAITING_KICK_COUNTDOWN";
                        this.countdownValue = 3;
                        this.countdownTimer = Date.now();
                    }
                }
            } else if (this.gameState === "WAITING_KICK") {
                // ── Live Aim Zone using biomechanical vectors ──────────────────
                if (joints['left_hip'] && joints['right_hip']) {
                    const lh = joints['left_hip'];
                    const rh = joints['right_hip'];
                    const ls = joints['left_shoulder'];
                    const rs = joints['right_shoulder'];
                    const la = joints['left_ankle'];
                    const ra = joints['right_ankle'];
                    const lk = joints['left_knee'];
                    const rk = joints['right_knee'];
                    
                    const bodyCenterX = (lh.x_norm + rh.x_norm) / 2.0;
                    const bodyCenterY = (lh.y_norm + rh.y_norm) / 2.0;
                    const shoulderCenterX = ls && rs ? (ls.x_norm + rs.x_norm) / 2.0 : bodyCenterX;
                    const bodyLean = shoulderCenterX - bodyCenterX;
                    this._lastBodyLean = bodyLean;
                    
                    // Dominant leg aim
                    const activeLeg = this.dominantLeg || 'right';
                    const activeAnkle = activeLeg === 'left' ? la : ra;
                    const activeHip   = activeLeg === 'left' ? lh : rh;
                    const activeKnee  = activeLeg === 'left' ? lk : rk;
                    
                    if (activeAnkle && activeHip) {
                        const legLength = this.calibrationData.legLength || 0.35;
                        
                        // Hip → knee → ankle vectors
                        const kneeX = activeKnee ? activeKnee.x_norm : (activeHip.x_norm + activeAnkle.x_norm) / 2;
                        const kneeY = activeKnee ? activeKnee.y_norm : (activeHip.y_norm + activeAnkle.y_norm) / 2;
                        const hipKneeVx = kneeX - activeHip.x_norm;
                        const kneeAnkleVx = activeAnkle.x_norm - kneeX;
                        const swingVx = hipKneeVx * 0.35 + kneeAnkleVx * 0.65;
                        
                        const shotHorizontal = swingVx * 2.8 + bodyLean * 1.2;
                        
                        let direction = "CENTER";
                        if (shotHorizontal < -0.08) direction = "LEFT";
                        else if (shotHorizontal > 0.08) direction = "RIGHT";
                        
                        const swingHeightRatio = Math.abs(activeHip.y_norm - activeAnkle.y_norm) / legLength;
                        const kneeAnkleVy = activeAnkle.y_norm - kneeY;
                        const upwardComponent = -kneeAnkleVy;
                        
                        let height = "MID";
                        if (swingHeightRatio < 0.62 || upwardComponent > 0.05) height = "TOP";
                        else if (swingHeightRatio > 0.92 && upwardComponent < -0.02) height = "BOTTOM";
                        
                        this.currentAimZone = `${height}_${direction}`;
                        this.updateGoalkeeperPerception();
                    }
                }
                
                // Live power bar from bioHistory velocity
                const activeVel = this.lastLeftVel > this.lastRightVel ? this.lastLeftVel : this.lastRightVel;
                const minVel = this.speedThreshold;
                const maxVel = this.speedThreshold * 3.2;
                const currentPower = Math.max(0, Math.min(100, Math.round(((activeVel - minVel) / (maxVel - minVel)) * 85 + 15)));
                this.updatePowerBar(currentPower);
                
                // Live HUD display for Target Estimation
                const promptEl = document.getElementById('game-prompt');
                if (promptEl) {
                    const parts = this.currentAimZone.split('_');
                    const heightStr = parts[0] === 'TOP' ? 'HIGH' : (parts[0] === 'BOTTOM' ? 'LOW' : 'MID');
                    const dirStr = parts[1] || 'CENTER';
                    promptEl.innerHTML = `Target: ${dirStr} | Height: ${heightStr} | Power: ${currentPower}%`;
                    if (!promptEl.classList.contains('active')) {
                        promptEl.classList.add('active');
                    }
                }
                
                this.checkKickEvent(joints);
            }
        } else if (this.state === "PLAY_GOALKEEPER") {
            // During GK inline calibration: collect neutral hand position samples
            if (this.gameState === "GK_CALIBRATION") {
                if (joints['left_wrist'] && joints['right_wrist']) {
                    const lw = joints['left_wrist'];
                    const rw = joints['right_wrist'];
                    this.gkNeutralSamples.push({
                        x: (lw.x_norm + rw.x_norm) / 2.0,
                        y: (lw.y_norm + rw.y_norm) / 2.0,
                        ly: lw.y_norm,
                        ry: rw.y_norm
                    });
                }
            } else {
                this.positionGloves(joints);
            }
        }
    }

    completeStrikerCalibration() {
        if (this.strikerCalibSamples.length === 0) {
            this.calibrationData.strikerNeutralHipX = 0.5;
            this.calibrationData.strikerNeutralHipY = 0.65;
            this.calibrationData.legLength = 0.35;
            this.strikerCalibrated = false;
            return;
        }
        const avg = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
        this.calibrationData.strikerNeutralHipX = avg(this.strikerCalibSamples.map(s => s.hipX));
        this.calibrationData.strikerNeutralHipY = avg(this.strikerCalibSamples.map(s => s.hipY));
        const leftLeg = avg(this.strikerCalibSamples.map(s => s.leftLegLen));
        const rightLeg = avg(this.strikerCalibSamples.map(s => s.rightLegLen));
        this.calibrationData.legLength = (leftLeg + rightLeg) / 2.0;
        
        this.strikerCalibrated = true;
        this.strikerCalibSamples = [];
        console.log("Striker calibration complete:", this.calibrationData);
    }

    updatePowerBar(val) {
        const bar = document.getElementById('power-bar');
        const txt = document.getElementById('power-text');
        if (bar && txt) {
            bar.style.width = `${val}%`;
            txt.innerText = `${val}%`;
        }
    }

    // ── BIOMECHANICAL KICK ANALYSIS SYSTEM ────────────────────────────────
    // Uses 6 joints: left/right hip, knee, ankle
    // 4 Phases: IDLE → BACKSWING → STRIKE → FOLLOW_THROUGH
    // Direction/height derived from hip→knee and knee→ankle vectors
    checkKickEvent(joints) {
        // Need at minimum hips and ankles; knees improve accuracy
        const la = joints['left_ankle'],  ra = joints['right_ankle'];
        const lk = joints['left_knee'],   rk = joints['right_knee'];
        const lh = joints['left_hip'],    rh = joints['right_hip'];
        const ls = joints['left_shoulder'], rs = joints['right_shoulder'];
        if (!la || !ra || !lh || !rh) return;
        
        const now = Date.now();
        const legLength = this.calibrationData.legLength || 0.35;
        const bodyCenterX = (lh.x_norm + rh.x_norm) / 2.0;
        const bodyCenterY = (lh.y_norm + rh.y_norm) / 2.0;
        
        // ── 1. Update ankle history (legacy, for velocity fallback) ──────────
        this.ankleHistory.left.push({ time: now, x: la.x_norm, y: la.y_norm });
        this.ankleHistory.right.push({ time: now, x: ra.x_norm, y: ra.y_norm });
        if (this.ankleHistory.left.length > this.historyLen) this.ankleHistory.left.shift();
        if (this.ankleHistory.right.length > this.historyLen) this.ankleHistory.right.shift();
        
        // ── 2. Update biomechanical history (full 6-joint frames) ────────────
        const bioFrame = {
            time: now,
            lhX: lh.x_norm, lhY: lh.y_norm,
            rhX: rh.x_norm, rhY: rh.y_norm,
            lkX: lk ? lk.x_norm : (lh.x_norm + la.x_norm) / 2,
            lkY: lk ? lk.y_norm : (lh.y_norm + la.y_norm) / 2,
            rkX: rk ? rk.x_norm : (rh.x_norm + ra.x_norm) / 2,
            rkY: rk ? rk.y_norm : (rh.y_norm + ra.y_norm) / 2,
            laX: la.x_norm, laY: la.y_norm,
            raX: ra.x_norm, raY: ra.y_norm
        };
        // Store per-leg history slices
        this.bioHistory.left.push({ time: now, hipX: lh.x_norm, hipY: lh.y_norm, kneeX: bioFrame.lkX, kneeY: bioFrame.lkY, ankleX: la.x_norm, ankleY: la.y_norm });
        this.bioHistory.right.push({ time: now, hipX: rh.x_norm, hipY: rh.y_norm, kneeX: bioFrame.rkX, kneeY: bioFrame.rkY, ankleX: ra.x_norm, ankleY: ra.y_norm });
        if (this.bioHistory.left.length > this.bioHistoryLen) this.bioHistory.left.shift();
        if (this.bioHistory.right.length > this.bioHistoryLen) this.bioHistory.right.shift();
        
        // ── 3. Auto-detect dominant kicking leg ─────────────────────────────
        // Leg that is raised higher (smaller y_norm) above hip is more likely the kicking leg
        const leftLegRaise  = bodyCenterY - la.y_norm; // positive = raised
        const rightLegRaise = bodyCenterY - ra.y_norm;
        // Weighted vote: strongly raised leg wins
        if (leftLegRaise > rightLegRaise + 0.04) {
            this.dominantLegVotes.left = Math.min(20, this.dominantLegVotes.left + 2);
            this.dominantLegVotes.right = Math.max(0, this.dominantLegVotes.right - 1);
        } else if (rightLegRaise > leftLegRaise + 0.04) {
            this.dominantLegVotes.right = Math.min(20, this.dominantLegVotes.right + 2);
            this.dominantLegVotes.left = Math.max(0, this.dominantLegVotes.left - 1);
        }
        this.dominantLeg = this.dominantLegVotes.left >= this.dominantLegVotes.right ? 'left' : 'right';
        
        // Select active leg joints
        const activeLeg = this.dominantLeg;
        const activeAnkle = activeLeg === 'left' ? la : ra;
        const activeKnee  = activeLeg === 'left' ? (lk || null) : (rk || null);
        const activeHip   = activeLeg === 'left' ? lh : rh;
        const activeBio   = this.bioHistory[activeLeg];
        
        // ── 4. Compute foot velocity (ankle movement speed) ───────────────────
        const calcVel = (hist) => {
            if (hist.length < 2) return 0;
            const a = hist[0], b = hist[hist.length - 1];
            const dt = (b.time - a.time) / 1000;
            if (dt <= 0) return 0;
            return Math.sqrt((b.ankleX-a.ankleX)**2 + (b.ankleY-a.ankleY)**2) / dt;
        };
        const leftVel  = calcVel(this.bioHistory.left);
        const rightVel = calcVel(this.bioHistory.right);
        const activeVel = activeLeg === 'left' ? leftVel : rightVel;
        this.lastLeftVel  = leftVel;
        this.lastRightVel = rightVel;
        
        // ── 5. Compute foot acceleration (for STRIKE detection) ────────────────
        let footAccel = 0;
        if (activeBio.length >= 3) {
            const n = activeBio.length;
            const v1 = Math.sqrt((activeBio[n-2].ankleX - activeBio[n-3].ankleX)**2 + (activeBio[n-2].ankleY - activeBio[n-3].ankleY)**2);
            const v2 = Math.sqrt((activeBio[n-1].ankleX - activeBio[n-2].ankleX)**2 + (activeBio[n-1].ankleY - activeBio[n-2].ankleY)**2);
            const dt = (activeBio[n-1].time - activeBio[n-3].time) / 1000;
            footAccel = dt > 0 ? (v2 - v1) / dt : 0;
        }
        this._liveFootAccel = footAccel;
        
        if (now - this.lastKickTime < this.coolDownTime) return;
        
        // ── 6. BIOMECHANICAL 4-PHASE STATE MACHINE ──────────────────────────
        if (this.kickPhase === "IDLE") {
            // Detect BACKSWING: kicking foot moves backward (y decreasing in screen coords = rising up)
            const legDist = Math.abs(activeHip.y_norm - activeAnkle.y_norm);
            if (legDist < legLength * 0.78) {
                // Foot is raised above natural standing position → entering backswing
                this.kickPhase = "BACKSWING";
                this.kickPhaseTimer = now;
                this.backswingAnchor = {
                    ankleX: activeAnkle.x_norm, ankleY: activeAnkle.y_norm,
                    hipX: activeHip.x_norm, hipY: activeHip.y_norm
                };
            }
        } else if (this.kickPhase === "BACKSWING") {
            if (now - this.kickPhaseTimer > 2500) {
                // Timed out, reset
                this.kickPhase = "IDLE";
                this.backswingAnchor = null;
                return;
            }
            // Wait for rapid forward acceleration (the actual strike)
            // Acceleration threshold scales with difficulty: harder = smaller threshold (more sensitive)
            const accelThresholds = { "EASY": 0.4, "MEDIUM": 0.28, "HARD": 0.18 };
            const accelThresh = accelThresholds[this.difficultyOptions[this.difficultyIndex]] || 0.28;
            if (footAccel > accelThresh || activeVel > this.speedThreshold * 1.4) {
                this.kickPhase = "STRIKE";
                this.strikeSnapshot = { ...activeBio[activeBio.length - 1] };
                this.kickPhaseTimer = now;
            }
        } else if (this.kickPhase === "STRIKE") {
            // Immediately transition to FOLLOW_THROUGH, fire the shot
            this.kickPhase = "FOLLOW_THROUGH";
            this.kickPhaseTimer = now;
            this.lastKickTime = now;
            this.kickTime = now;
            this.audio.playSound('kick');
            
            // ── BIOMECHANICAL SHOT ANALYSIS ────────────────────────────────────
            // Use FULL KINEMATIC CHAIN: hip → knee → ankle vectors
            
            // Vector A: hip → knee
            const hipKneeVx = (activeKnee ? activeKnee.x_norm : (activeHip.x_norm + activeAnkle.x_norm) / 2) - activeHip.x_norm;
            const hipKneeVy = (activeKnee ? activeKnee.y_norm : (activeHip.y_norm + activeAnkle.y_norm) / 2) - activeHip.y_norm;
            
            // Vector B: knee → ankle
            const kneeAx = activeKnee ? activeKnee.x_norm : (activeHip.x_norm + activeAnkle.x_norm) / 2;
            const kneeAy = activeKnee ? activeKnee.y_norm : (activeHip.y_norm + activeAnkle.y_norm) / 2;
            const kneeAnkleVx = activeAnkle.x_norm - kneeAx;
            const kneeAnkleVy = activeAnkle.y_norm - kneeAy;
            
            // Combined swing direction vector (weighted: ankle segment carries more weight)
            const swingVx = hipKneeVx * 0.35 + kneeAnkleVx * 0.65;
            const swingVy = hipKneeVy * 0.35 + kneeAnkleVy * 0.65;
            
            // Body orientation from shoulders
            const shoulderDx = ls && rs ? (ls.x_norm - rs.x_norm) : 0;
            const shoulderDy = ls && rs ? (ls.y_norm - rs.y_norm) : 0;
            const bodyOrient = (Math.atan2(shoulderDy, shoulderDx) * 180) / Math.PI;
            const bodyLean = ls && rs ? ((ls.x_norm + rs.x_norm) / 2) - bodyCenterX : 0;
            
            // ── HORIZONTAL SHOT DIRECTION ───────────────────────────────────
            // Combine: swing vector X + body lean + ankle history trajectory
            let ankleMotionX = 0;
            if (activeBio.length >= 2) {
                ankleMotionX = activeBio[activeBio.length - 1].ankleX - activeBio[0].ankleX;
            }
            // Backswing anchor tells us where foot was before swing
            const backswingDriftX = this.backswingAnchor
                ? (activeAnkle.x_norm - this.backswingAnchor.ankleX)
                : 0;
            
            // Weighted combination for horizontal target
            const shotHorizontal = (
                swingVx * 2.8 +
                ankleMotionX * 2.0 +
                backswingDriftX * 1.5 +
                bodyLean * 1.2
            );
            
            let direction = "CENTER";
            let dirLabel = 1;
            const hThresh = 0.08; // tuned: lower = more sensitive
            if (shotHorizontal < -hThresh) {
                direction = "LEFT"; dirLabel = 0;
            } else if (shotHorizontal > hThresh) {
                direction = "RIGHT"; dirLabel = 2;
            }
            
            // ── VERTICAL SHOT HEIGHT ────────────────────────────────────────
            // Primary: how high is ankle relative to hip (swing height)
            const swingHeightRatio = Math.abs(activeHip.y_norm - activeAnkle.y_norm) / legLength;
            // Secondary: upward swing component (positive Y vector in screen coords = downward)
            // swingVy < 0 means foot is swinging upward → top shot
            const upwardComponent = -swingVy; // positive = upward swing
            
            let height = "MID";
            if (swingHeightRatio < 0.62 || upwardComponent > 0.05) {
                height = "TOP";
            } else if (swingHeightRatio > 0.92 && upwardComponent < -0.02) {
                height = "BOTTOM";
            }
            
            const targetZone = `${height}_${direction}`;
            
            // ── POWER: proportional to foot velocity + acceleration ──────────
            const minVel = this.speedThreshold;
            const maxVel = this.speedThreshold * 3.2;
            const velPower = Math.max(0, Math.min(1, (activeVel - minVel) / (maxVel - minVel)));
            const accelBonus = Math.max(0, Math.min(0.2, footAccel * 0.08));
            const power = Math.max(30, Math.min(100, Math.round((velPower + accelBonus) * 70 + 30)));
            this.updatePowerBar(power);
            
            // ── SIDE SPIN / CURVE (from strike angle) ───────────────────────
            // Angle between knee-ankle vector and vertical axis
            const kickAngle = activeKnee
                ? (Math.atan2(activeAnkle.x_norm - activeKnee.x_norm,
                              activeAnkle.y_norm - activeKnee.y_norm) * 180) / Math.PI
                : 0.0;
            
            // ── AI SHOT PREDICTOR ────────────────────────────────────────────
            const predictionResult = this.predictor.predict(kickAngle, activeVel, bodyOrient);
            const dirNames = ["LEFT", "CENTER", "RIGHT"];
            const predictedDirection = dirNames[predictionResult.label];
            this.predictor.trainSample(kickAngle, activeVel, bodyOrient, dirLabel);
            
            this.kickInfo = {
                leg: activeLeg,
                velocity: activeVel,
                acceleration: footAccel,
                power: power,
                direction: direction,
                height: height,
                targetZone: targetZone,
                kick_angle: kickAngle,
                body_orientation: bodyOrient,
                swingVx: swingVx,
                swingVy: swingVy,
                prediction: predictedDirection,
                confidence: predictionResult.confidence
            };
            
            // Store for training overlay
            this._biomechSnapshot = {
                hipKneeVx, hipKneeVy, kneeAnkleVx, kneeAnkleVy, swingVx, swingVy,
                swingHeightRatio, upwardComponent, shotHorizontal, activeLeg
            };
            
            this.ballTrajectory = this.physics.calculateTrajectory(targetZone, power, kickAngle);
            this.trajectoryIndex = 0;
            this.determineAiGoalkeeperDive(targetZone);
            
            const promptEl = document.getElementById('game-prompt');
            if (promptEl) promptEl.classList.remove('active');
            this.gameState = "BALL_FLIGHT";
            
        } else if (this.kickPhase === "FOLLOW_THROUGH") {
            // Stay in follow-through for 200ms then reset
            if (now - this.kickPhaseTimer > 200) {
                this.kickPhase = "IDLE";
                this.backswingAnchor = null;
            }
        }
    }

    determineAiGoalkeeperDive(zone) {
        const diff = this.difficultyOptions[this.difficultyIndex] || "MEDIUM";
        
        // Define frame-based reaction delays relative to total ball flight frames
        // Easy: reacts at 60% of flight (has 40% time to dive)
        // Medium: reacts at 40% of flight (has 60% time to dive)
        // Hard: reacts at 20% of flight (has 80% time to dive)
        const delayRatios = { "EASY": 0.60, "MEDIUM": 0.40, "HARD": 0.20 };
        const delayRatio = delayRatios[diff] || 0.40;
        this.aiDiveTriggerIndex = Math.floor(this.ballTrajectory.length * delayRatio);

        // If keeper predicted target was not established (e.g. mouse fallback), run a direct prediction roll now
        if (!this.keeperPredictedTarget) {
            // Target prediction accuracy rates matching user scoring chances:
            // Easy: 80% score chance -> 20% correct prediction
            // Medium: 50% score chance -> 50% correct prediction
            // Hard: 20% score chance -> 80% correct prediction
            const accuracyRates = { "EASY": 0.20, "MEDIUM": 0.50, "HARD": 0.80 };
            const accuracy = accuracyRates[diff];
            const correct = Math.random() < accuracy;
            
            if (correct) {
                this.keeperPredictedTarget = zone;
            } else {
                const adjacents = this.getAdjacentZones(zone);
                if (adjacents.length > 0) {
                    this.keeperPredictedTarget = adjacents[Math.floor(Math.random() * adjacents.length)];
                } else {
                    this.keeperPredictedTarget = zone;
                }
            }
            
            const baseConfRange = { "EASY": [30, 50], "MEDIUM": [60, 75], "HARD": [80, 95] };
            const range = baseConfRange[diff];
            this.keeperPredictionConfidence = Math.round(range[0] + Math.random() * (range[1] - range[0]));
        }

        // The keeper reacts to their locked prediction, not perfect information!
        const predZone = this.keeperPredictedTarget;
        
        let horiz = "CENTER";
        if (predZone.endsWith("LEFT")) horiz = "LEFT";
        else if (predZone.endsWith("RIGHT")) horiz = "RIGHT";
        
        let vert = "MID";
        if (predZone.startsWith("TOP")) vert = "HIGH";
        else if (predZone.startsWith("BOTTOM")) vert = "LOW";
        
        this.keeperDiveDir = horiz;
        this.keeperDiveHeight = vert;

        const gl  = this.physics.strikerGoalLeft;
        const gr  = this.physics.strikerGoalRight;
        const gt  = this.physics.strikerGoalTop;
        const gb  = this.physics.strikerGoalBottom;
        const gw  = gr - gl;
        const gh  = gb - gt;
        const gcx = this.physics.goalCenterX;
        
        // ── Set goalkeeper anchor position (body center for animation) ──────
        if (this.keeperDiveDir === "LEFT") {
            this.keeperTargetX = gl + Math.floor(gw * 0.30);
        } else if (this.keeperDiveDir === "RIGHT") {
            this.keeperTargetX = gr - Math.floor(gw * 0.30);
        } else {
            this.keeperTargetX = gcx;
        }
        if (this.keeperDiveHeight === "HIGH") {
            this.keeperTargetY = gb - Math.floor(gh * 0.32);
        } else if (this.keeperDiveHeight === "LOW") {
            this.keeperTargetY = gb - Math.floor(gh * 0.07);
        } else {
            this.keeperTargetY = gb - Math.floor(gh * 0.20);
        }
        
        // ── Build physical colliders (head, torso, left arm, right arm) ──────
        // Collider positions are computed from final keeperTarget position.
        // Difficulty scales arm reach width.
        const reachScale = { "EASY": 0.12, "MEDIUM": 0.16, "HARD": 0.22 };
        const reach = (reachScale[this.difficultyOptions[this.difficultyIndex]] || 0.16) * gw;
        const keepH = Math.floor(gh * 0.32); // goalkeeper avatar height (~32% of goal)
        
        // Body parts relative to final target position (will be evaluated at ball arrival)
        this._keeperColliderDef = [
            // Head
            { id: 'head',       ox: 0,        oy: -keepH * 0.45, w: keepH * 0.22, h: keepH * 0.22 },
            // Torso
            { id: 'torso',      ox: 0,        oy: -keepH * 0.15, w: keepH * 0.42, h: keepH * 0.36 },
            // Left arm (extends further when diving left)
            { id: 'left_arm',   ox: -(reach + keepH * 0.18), oy: -keepH * 0.25, w: reach, h: keepH * 0.16 },
            // Right arm
            { id: 'right_arm',  ox: keepH * 0.18,             oy: -keepH * 0.25, w: reach, h: keepH * 0.16 }
        ];
    }

    getAdjacentZones(zone) {
        const parts = zone.split('_');
        if (parts.length < 2) return [];
        const h = parts[0]; // TOP, MID, BOTTOM
        const d = parts[1]; // LEFT, CENTER, RIGHT
        
        const heights = ['BOTTOM', 'MID', 'TOP'];
        const dirs = ['LEFT', 'CENTER', 'RIGHT'];
        
        const hIdx = heights.indexOf(h);
        const dIdx = dirs.indexOf(d);
        if (hIdx === -1 || dIdx === -1) return [];
        
        const adjacents = [];
        for (let dh = -1; dh <= 1; dh++) {
            for (let dd = -1; dd <= 1; dd++) {
                if (dh === 0 && dd === 0) continue;
                const nh = hIdx + dh;
                const nd = dIdx + dd;
                if (nh >= 0 && nh < heights.length && nd >= 0 && nd < dirs.length) {
                    adjacents.push(`${heights[nh]}_${dirs[nd]}`);
                }
            }
        }
        return adjacents;
    }

    updateGoalkeeperPerception() {
        const now = Date.now();
        if (now - this.lastPerceptionUpdateTime < 300) return;
        this.lastPerceptionUpdateTime = now;

        const diff = this.difficultyOptions[this.difficultyIndex] || "MEDIUM";
        
        // Target prediction accuracy rates matching user scoring chances:
        // Easy: 80% score chance -> 20% correct prediction
        // Medium: 50% score chance -> 50% correct prediction
        // Hard: 20% score chance -> 80% correct prediction
        const accuracyRates = { "EASY": 0.20, "MEDIUM": 0.50, "HARD": 0.80 };
        const accuracy = accuracyRates[diff];
        
        const roll = Math.random();
        const correct = roll < accuracy;
        
        let predicted = this.currentAimZone;
        if (!correct) {
            // Find adjacent zones
            const adjacents = this.getAdjacentZones(this.currentAimZone);
            if (adjacents.length > 0) {
                predicted = adjacents[Math.floor(Math.random() * adjacents.length)];
            }
        }
        
        this.keeperPredictedTarget = predicted;
        
        // Calculate dynamic confidence based on body stability and difficulty
        const baseConfRange = {
            "EASY": [30, 50],
            "MEDIUM": [60, 75],
            "HARD": [80, 95]
        };
        const range = baseConfRange[diff];
        let conf = range[0] + Math.random() * (range[1] - range[0]);
        
        // Stability factor: active foot velocity. High velocity = less confident guess
        const activeLeg = this.dominantLeg || 'right';
        const activeVel = activeLeg === 'left' ? this.lastLeftVel : this.lastRightVel;
        if (activeVel > this.speedThreshold) {
            const speedFactor = Math.min(1.0, (activeVel - this.speedThreshold) / (this.speedThreshold * 2.0));
            conf -= speedFactor * 15;
        }
        
        // Body lean factor: strong body lean makes direction clearer, boosting confidence
        if (this._lastBodyLean) {
            const leanFactor = Math.min(1.0, Math.abs(this._lastBodyLean) / 0.15);
            conf += leanFactor * 8;
        }
        
        this.keeperPredictionConfidence = Math.max(10, Math.min(99, Math.round(conf)));
    }

    // STATE LOOP: STRIKER MODE
    runStrikerLoop() {
        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // ── STRIKER INLINE CALIBRATION OVERLAY ────────────────────────────
        if (this.gameState === "STRIKER_CALIBRATION") {
            this.drawStrikerScene(ctx);
            this.drawFirstPersonGoal(ctx);
            this.drawFirstPersonGoalkeeper(ctx);
            
            ctx.save();
            ctx.fillStyle = 'rgba(0, 0, 0, 0.75)';
            ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
            
            // Neon accent border
            ctx.strokeStyle = '#64ff64';
            ctx.lineWidth = 4;
            ctx.strokeRect(40, 40, this.canvas.width - 80, this.canvas.height - 80);
            
            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 32px Outfit';
            ctx.textAlign = 'center';
            ctx.shadowBlur = 15;
            ctx.shadowColor = '#64ff64';
            ctx.fillText('STRIKER CALIBRATION', 512, 220);
            ctx.shadowBlur = 0;
            
            ctx.font = '20px Outfit';
            ctx.fillStyle = '#e0e0e0';
            ctx.fillText('Stand naturally in the center of the camera view.', 512, 280);
            ctx.fillStyle = '#ffe650';
            ctx.fillText('Hold a natural, ready posture to measure leg length and hip level...', 512, 315);
            
            // Draw progress circle
            const elapsed = Date.now() - this.strikerCalibPhaseTimer;
            const pct = Math.min(1.0, elapsed / 3000);
            
            ctx.beginPath();
            ctx.arc(512, 420, 50, 0, 2 * Math.PI);
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
            ctx.lineWidth = 8;
            ctx.stroke();
            
            ctx.beginPath();
            ctx.arc(512, 420, 50, -Math.PI / 2, -Math.PI / 2 + 2 * Math.PI * pct);
            ctx.strokeStyle = '#64ff64';
            ctx.lineWidth = 8;
            ctx.lineCap = 'round';
            ctx.stroke();
            
            ctx.font = 'bold 24px Outfit';
            ctx.fillStyle = '#ffffff';
            ctx.fillText(Math.ceil(3 - elapsed / 1000).toString(), 512, 429);
            ctx.restore();
            
            this.drawCornersWebcam();
            this.updateBroadcastOverlay();
            this.updateDebugPanel();
            this.updateHUDBar();
            return;
        }
        
        // ── REGULAR GAMEPLAY RENDER ───────────────────────────────────────
        // 1. Draw immersive stadium
        this.drawStrikerScene(ctx);
        
        // 2. Draw goal net
        this.drawFirstPersonGoal(ctx);
        
        // 3. Draw aiming reticle (if waiting for kick)
        if (this.gameState === "WAITING_KICK" || this.gameState === "WAITING_KICK_COUNTDOWN") {
            this.drawAimReticle(ctx, this.currentAimZone);
        }
        
        // Handle Countdown Sequence
        if (this.gameState === "WAITING_KICK_COUNTDOWN") {
            const elapsed = Date.now() - this.countdownTimer;
            if (elapsed > 1000) {
                this.countdownValue--;
                this.countdownTimer = Date.now();
                
                const cdEl = document.getElementById('game-countdown');
                if (this.countdownValue > 0) {
                    if (cdEl) cdEl.innerText = this.countdownValue.toString();
                    this.audio.playSound('kick');
                } else if (this.countdownValue === 0) {
                    if (cdEl) cdEl.innerText = "KICK!";
                    this.audio.playSound('save'); // Whistle
                    const promptEl = document.getElementById('game-prompt');
                    if (promptEl) promptEl.innerText = "KICK THE BALL!";
                } else {
                    if (cdEl) {
                        cdEl.style.display = 'none';
                        cdEl.classList.remove('active');
                    }
                    this.gameState = "WAITING_KICK";
                    this.ankleHistory.left = [];
                    this.ankleHistory.right = [];
                }
            }
        }
        
        // Update components based on sub-state
        if (this.gameState === "BALL_FLIGHT") {
            if (this.trajectoryIndex < this.ballTrajectory.length) {
                const pt = this.ballTrajectory[this.trajectoryIndex];
                this.ballScreenPos = { x: pt.x, y: pt.y };
                this.ballRadius = pt.radius;
                this.ballPos3d = [pt.x_3d, pt.y_3d, pt.z];
                
                // Track ball history trail
                this.ballHistory.push({ x: pt.x, y: pt.y, r: pt.radius });
                if (this.ballHistory.length > 8) this.ballHistory.shift();
                
                // Spin animation
                this.ballSpinAngle = (this.ballSpinAngle || 0) + 0.18;
                
                // Animate goalkeeper dive smoothly with frame-synced reaction delay and AI commitment
                const triggerIdx = this.aiDiveTriggerIndex || 0;
                const totalFrames = this.ballTrajectory.length;
                
                if (this.trajectoryIndex <= triggerIdx) {
                    // Keeper is in anticipation phase (weight shift, crouch)
                    this.keeperAiState = "ANTICIPATING";
                    const progress = triggerIdx > 0 ? this.trajectoryIndex / triggerIdx : 0.0;
                    let leanX = 0;
                    let leanY = 0;
                    let leanRot = 0;
                    
                    if (this.keeperDiveDir === "LEFT") {
                        leanX = -22 * progress;
                        leanY = 12 * progress; // slight crouch
                        leanRot = 8 * progress;
                    } else if (this.keeperDiveDir === "RIGHT") {
                        leanX = 22 * progress;
                        leanY = 12 * progress; // slight crouch
                        leanRot = -8 * progress;
                    } else {
                        leanY = 14 * progress; // crouch in center
                    }
                    
                    this.keeperX = 512 + leanX;
                    this.keeperY = this.physics.strikerGoalBottom + leanY;
                    this.keeperRotation = leanRot;
                } else {
                    // Reaction delay has passed -> Commit to dive
                    if (!this.keeperReactionStarted) {
                        this.keeperReactionStarted = true;
                        this.keeperCommitTime = Date.now();
                        this.keeperAiState = "DIVING";
                        this.keeperDiveStartX = this.keeperX;
                        this.keeperDiveStartY = this.keeperY;
                        this.keeperDiveStartRot = this.keeperRotation;
                    }
                    
                    const diveFrames = totalFrames - triggerIdx;
                    const elapsedDive = this.trajectoryIndex - triggerIdx;
                    const t = diveFrames > 0 ? elapsedDive / diveFrames : 1.0;
                    const tClamped = Math.max(0, Math.min(1.0, t));
                    const easeOutCubic = 1 - Math.pow(1 - tClamped, 3);
                    
                    this.keeperX = this.keeperDiveStartX + (this.keeperTargetX - this.keeperDiveStartX) * easeOutCubic;
                    
                    let targetRotation = 0;
                    if (this.keeperDiveDir === "LEFT") targetRotation = 45;
                    else if (this.keeperDiveDir === "RIGHT") targetRotation = -45;
                    
                    this.keeperRotation = this.keeperDiveStartRot + (targetRotation - this.keeperDiveStartRot) * easeOutCubic;
                    
                    // Arc trajectory for Y (dive extension + landing)
                    if (tClamped > 0.8) {
                        const landProgress = (tClamped - 0.8) / 0.2; // 0 to 1
                        this.keeperY = this.keeperTargetY + (this.physics.strikerGoalBottom - this.keeperTargetY) * landProgress * 0.45;
                        this.keeperAiState = "LANDED";
                    } else {
                        this.keeperY = this.keeperDiveStartY + (this.keeperTargetY - this.keeperDiveStartY) * easeOutCubic;
                        this.keeperAiState = "DIVING";
                    }
                }
                
                // Check mid-flight collision when ball is close enough in depth
                if (pt.z >= 0.82) {
                    const saveHit = this.checkGoalkeeperCollision(pt.x, pt.y, pt.radius);
                    if (saveHit) {
                        this.outcomeText = saveHit;
                        this.outcomeColor = "#32c8ff";
                        
                        this.gameState = "RESULT_CELEBRATION";
                        this.gameStateTimer = Date.now();
                        this.hasPlayedOutcomeSound = false;
                        this.strikerFlashTimer = 1.0;
                        
                        // Truncate ball trajectory so it stops at the save point
                        this.ballTrajectory = this.ballTrajectory.slice(0, this.trajectoryIndex + 1);
                    } else {
                        this.trajectoryIndex++;
                    }
                } else {
                    this.trajectoryIndex++;
                }
            } else {
                // Ball flight finished: evaluate outcome
                this.evaluateStrikerOutcome();
                this.gameState = "RESULT_CELEBRATION";
                this.gameStateTimer = Date.now();
                this.hasPlayedOutcomeSound = false;
                
                // Initialize screen flash intensity (1.0 = fully opaque)
                this.strikerFlashTimer = 1.0;
            }
        } else if (this.gameState === "RESULT_CELEBRATION") {
            if (!this.hasPlayedOutcomeSound) {
                if (this.outcomeText.includes("GOAL")) {
                    this.audio.playSound('goal');
                    this.audio.playSound('cheer');
                } else if (this.outcomeText.includes("SAVE")) {
                    this.audio.playSound('save');
                } else {
                    this.audio.playSound('miss');
                }
                this.hasPlayedOutcomeSound = true;
                
                // Show announcement
                const outcomeEl = document.getElementById('game-outcome');
                outcomeEl.innerText = this.outcomeText;
                outcomeEl.className = "game-announcement outcome-announcement active " + 
                    (this.outcomeText.includes("GOAL") ? "text-green" : (this.outcomeText.includes("SAVE") ? "text-blue" : "text-red"));
                
                const promptEl = document.getElementById('game-prompt');
                promptEl.innerText = `Power: ${this.kickInfo.power}% | Dir: ${this.kickInfo.direction} | Height: ${this.kickInfo.height}`;
                promptEl.classList.add('active');
            }
            
            if (Date.now() - this.gameStateTimer > 2500) {
                const isGoal = this.outcomeText.includes("GOAL");
                this.attempts++;
                if (isGoal) this.goals++;
                
                // Clear outcomes & trigger direct transition
                document.getElementById('game-outcome').classList.remove('active');
                document.getElementById('game-prompt').classList.remove('active');
                this.completeAttempt();
            }
        }
        
        // Draw predicted target marker, trajectory line, and ball trail (first 500ms of kick or if trainingMode)
        if (this.gameState === "BALL_FLIGHT" && this.ballTrajectory.length > 0) {
            const timeSinceKick = Date.now() - this.lastKickTime;
            if (timeSinceKick < 500 || this.trainingMode || this.debugMode) {
                const finalPt = this.ballTrajectory[this.ballTrajectory.length - 1];
                
                // 1. Draw predicted target marker (Crosshair)
                ctx.save();
                ctx.strokeStyle = 'rgba(255, 50, 150, 0.8)';
                ctx.lineWidth = 3;
                ctx.setLineDash([4, 4]);
                ctx.shadowBlur = 10;
                ctx.shadowColor = 'rgba(255, 50, 150, 0.6)';
                ctx.beginPath();
                ctx.arc(finalPt.x, finalPt.y, 18, 0, 2 * Math.PI);
                ctx.stroke();
                
                ctx.beginPath();
                ctx.moveTo(finalPt.x - 25, finalPt.y);
                ctx.lineTo(finalPt.x + 25, finalPt.y);
                ctx.moveTo(finalPt.x, finalPt.y - 25);
                ctx.lineTo(finalPt.x, finalPt.y + 25);
                ctx.stroke();
                ctx.restore();
                
                // 2. Draw trajectory dotted curve
                ctx.save();
                ctx.strokeStyle = 'rgba(255, 230, 80, 0.3)';
                ctx.lineWidth = 3;
                ctx.setLineDash([6, 6]);
                ctx.beginPath();
                ctx.moveTo(this.ballTrajectory[0].x, this.ballTrajectory[0].y);
                for (let i = 1; i < this.ballTrajectory.length; i++) {
                    ctx.lineTo(this.ballTrajectory[i].x, this.ballTrajectory[i].y);
                }
                ctx.stroke();
                ctx.restore();
            }
            
            // 3. Draw ball trail (solid white path behind ball)
            if (this.trajectoryIndex > 1) {
                ctx.save();
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
                ctx.lineWidth = 4;
                ctx.lineCap = 'round';
                ctx.beginPath();
                ctx.moveTo(this.ballTrajectory[0].x, this.ballTrajectory[0].y);
                for (let i = 1; i < this.trajectoryIndex; i++) {
                    ctx.lineTo(this.ballTrajectory[i].x, this.ballTrajectory[i].y);
                }
                ctx.stroke();
                ctx.restore();
            }
        }
        
        // 4. Draw goalie
        this.drawFirstPersonGoalkeeper(ctx);
        
        // 5. Draw ball
        if (this.gameState === "WAITING_KICK" || this.gameState === "WAITING_KICK_COUNTDOWN") {
            this.drawStrikerBallAtRest(ctx);
        } else {
            this.drawBall();
        }
        
        // Render prediction text if available
        if (this.gameState === "BALL_FLIGHT" && this.kickInfo) {
            ctx.fillStyle = '#ffe650';
            ctx.font = '16px Outfit';
            ctx.textAlign = 'center';
            ctx.shadowBlur = 10;
            ctx.shadowColor = 'rgba(255, 230, 80, 0.5)';
            ctx.fillText(`AI SHOT PREDICTION: ${this.kickInfo.prediction} (${Math.round(this.kickInfo.confidence * 100)}% Conf)`, 512, 85);
            ctx.shadowBlur = 0;
        }
        
        // ── GOAL/MISS SCREEN FLASH EFFECT ─────────────────────────────────
        if (this.strikerFlashTimer > 0) {
            ctx.save();
            const color = this.outcomeText.includes("GOAL") ? "46, 204, 113" : "231, 76, 60";
            ctx.fillStyle = `rgba(${color}, ${this.strikerFlashTimer * 0.45})`;
            ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
            ctx.restore();
            this.strikerFlashTimer -= 0.025;
        }
        
        // ── TRAINING MODE OVERLAY ──────────────────────────────────────────
        if (this.trainingMode) {
            this.drawTrainingOverlay(ctx);
        }
        
        // Render corners camera preview
        this.drawCornersWebcam();
        
        // Update Live Broadcast elements & debug panels
        this.updateBroadcastOverlay();
        this.updateDebugPanel();
        
        // Update top HUD bar values
        this.updateHUDBar();
    }

    evaluateStrikerOutcome() {
        // Final fallback collision check at z=1
        const ballX = this.ballScreenPos.x;
        const ballY = this.ballScreenPos.y;
        const ballR = Math.max(6, this.physics.endRadius);
        
        // ── DETERMINISTIC SCREEN-SPACE GOAL DETECTION ───────────────────
        const gl = this.physics.strikerGoalLeft;
        const gr = this.physics.strikerGoalRight;
        const gt = this.physics.strikerGoalTop;
        const gb = this.physics.strikerGoalBottom;
        
        const inside = (ballX >= gl && ballX <= gr && ballY >= gt && ballY <= gb);
        
        if (!inside) {
            this.outcomeText = "MISS! WIDE / HIGH";
            this.outcomeColor = "#ff6464";
            return;
        }
        
        // Final fallback collision check at z=1
        const saveHit = this.checkGoalkeeperCollision(ballX, ballY, ballR);
        if (saveHit) {
            this.outcomeText = saveHit;
            this.outcomeColor = "#32c8ff";
            return;
        }
        
        this.outcomeText = "GOAL!!!";
        this.outcomeColor = "#64ff64";
    }

    checkGoalkeeperCollision(ballX, ballY, ballR) {
        const def = this._keeperColliderDef;
        if (def && def.length > 0) {
            const kx = this.keeperX;
            const ky = this.keeperY;
            const rot = this.keeperRotation || 0; // in degrees
            const rotRad = -rot * Math.PI / 180.0;
            
            // Translate ball relative to goalkeeper pivot, then rotate it back
            const dx = ballX - kx;
            const dy = ballY - ky;
            const localBallX = dx * Math.cos(rotRad) - dy * Math.sin(rotRad);
            const localBallY = dx * Math.sin(rotRad) + dy * Math.cos(rotRad);
            
            this.keeperColliders = def.map(c => ({
                id: c.id,
                localX: c.ox,
                localY: c.oy,
                w: c.w,
                h: c.h,
                x: kx + c.ox - c.w / 2,
                y: ky + c.oy - c.h / 2
            }));
            
            for (const col of def) {
                // Local rectangle bounds
                const left = col.ox - col.w / 2;
                const right = col.ox + col.w / 2;
                const top = col.oy - col.h / 2;
                const bottom = col.oy + col.h / 2;
                
                // Find closest point on unrotated local rectangle to the rotated ball position
                const cx = Math.max(left, Math.min(right, localBallX));
                const cy = Math.max(top, Math.min(bottom, localBallY));
                
                const dist = Math.sqrt((localBallX - cx) ** 2 + (localBallY - cy) ** 2);
                if (dist <= ballR + 4) {
                    const saveLabels = { head: 'FINGERTIP SAVE!', torso: 'BODY SAVE!', left_arm: 'DIVING SAVE!', right_arm: 'DIVING SAVE!' };
                    return saveLabels[col.id] || "SAVED BY KEEPER!";
                }
            }
        } else {
            // Fallback: simple radius check (legacy)
            const saveRadii = { "EASY": 75, "MEDIUM": 90, "HARD": 115 };
            const saveR = saveRadii[this.difficultyOptions[this.difficultyIndex]] || 90;
            const dist = Math.sqrt((ballX - this.keeperX) ** 2 + (ballY - this.keeperY) ** 2);
            if (dist < saveR && this.keeperDiveDir === (this.kickInfo && this.kickInfo.direction)) {
                return "SAVED BY KEEPER!";
            }
        }
        return null;
    }

    positionGloves(joints) {
        const isGK = (this.gameModeType === "GOALKEEPER");
        const goalW = isGK ? this.physics.gkGoalWidth : this.physics.goalWidth;
        const goalH = isGK ? this.physics.gkGoalHeight : this.physics.goalHeight;
        const goalL = isGK ? this.physics.gkGoalLeft : this.physics.goalLeft;
        const goalR = isGK ? this.physics.gkGoalRight : this.physics.goalRight;
        const goalT = isGK ? this.physics.gkGoalTop : this.physics.goalTop;
        const goalB = isGK ? this.physics.gkGoalBottom : this.physics.goalBottom;
        const goalCX = (goalL + goalR) / 2;
        const goalCY = (goalT + goalB) / 2;
        
        // ── Neutral calibration values ────────────────────────────────────
        const neutralX  = this.calibrationData.gkNeutralX  || 0.5;
        const neutralY  = this.calibrationData.gkNeutralY  || 0.45;
        const reachTop  = this.calibrationData.gkReachTop  || 0.18;
        const reachBott = this.calibrationData.gkReachBottom || 0.72;
        const reachSpan = Math.max(0.01, reachBott - reachTop);
        
        // ── Sensitivity (amplify vertical more than horizontal) ───────────
        const X_SENS = isGK ? 2.5 : 1.8;  // horizontal amplification
        const Y_SENS = isGK ? 4.0 : 2.5;  // vertical amplification
        
        // ── Dead zone (normalized coords) ────────────────────────────────
        const DEAD_X = 0.015;
        const DEAD_Y = 0.012;
        
        const applyDead = (delta, threshold) => {
            if (Math.abs(delta) < threshold) return 0;
            return delta > 0 ? delta - threshold : delta + threshold;
        };
        
        // ── Per-wrist mapping helper ──────────────────────────────────────
        const mapWrist = (wrist) => {
            if (!wrist) return null;
            
            let dx = wrist.x_norm - neutralX;
            let dy = wrist.y_norm - neutralY; // positive = downward in normalized coords
            
            // Apply dead zone
            dx = applyDead(dx, DEAD_X);
            dy = applyDead(dy, DEAD_Y);
            
            let screenX, screenY;
            
            if (isGK) {
                // ── GOALKEEPER: goal-reach interpolation + delta amplification ──
                // Horizontal: delta from neutral center
                screenX = goalCX + dx * X_SENS * goalW;
                
                // Vertical: map absolute wrist_y into goal using reach calibration
                // Then layer on top a delta boost for responsiveness
                const normY = (wrist.y_norm - reachTop) / reachSpan; // 0 = top, 1 = bottom
                const interpY = goalT + Math.max(0, Math.min(1, normY)) * goalH;
                
                // Blend interpolated and delta-boosted positions
                const deltaY = dy * Y_SENS * goalH;
                screenY = interpY * 0.6 + (goalCY + deltaY) * 0.4;
            } else {
                // ── STRIKER MODE: simpler delta-only mapping ──────────────────
                screenX = goalCX + dx * X_SENS * goalW;
                screenY = goalCY + dy * Y_SENS * goalH;
            }
            
            return { x: Math.round(screenX), y: Math.round(screenY) };
        };
        
        const lPos = mapWrist(joints['left_wrist']);
        const rPos = mapWrist(joints['right_wrist']);
        
        // Clamp within extended goal area
        const marginX = isGK ? 130 : 80;
        const marginY = isGK ? 100 : 60;
        const clampX = (x) => Math.max(goalL - marginX, Math.min(goalR + marginX, x));
        const clampY = (y) => Math.max(goalT - marginY, Math.min(goalB + marginY, y));
        
        if (lPos) {
            this.leftGlovePos.x = clampX(lPos.x);
            this.leftGlovePos.y = clampY(lPos.y);
        }
        if (rPos) {
            this.rightGlovePos.x = clampX(rPos.x);
            this.rightGlovePos.y = clampY(rPos.y);
        }
        
        // Store raw/mapped data for debug overlay
        this._gkDebug = {
            rawLX: joints['left_wrist']  ? joints['left_wrist'].x_norm.toFixed(3)  : '-',
            rawLY: joints['left_wrist']  ? joints['left_wrist'].y_norm.toFixed(3)  : '-',
            rawRX: joints['right_wrist'] ? joints['right_wrist'].x_norm.toFixed(3) : '-',
            rawRY: joints['right_wrist'] ? joints['right_wrist'].y_norm.toFixed(3) : '-',
            mapLX: lPos ? lPos.x : '-',
            mapLY: lPos ? lPos.y : '-',
            mapRX: rPos ? rPos.x : '-',
            mapRY: rPos ? rPos.y : '-',
            neutralX: neutralX.toFixed(3),
            neutralY: neutralY.toFixed(3),
            reachTop: reachTop.toFixed(3),
            reachBott: reachBott.toFixed(3),
            xSens: X_SENS,
            ySens: Y_SENS,
            reachPctL: joints['left_wrist']  ? (((joints['left_wrist'].y_norm  - reachTop) / reachSpan) * 100).toFixed(0) + '%' : '-',
            reachPctR: joints['right_wrist'] ? (((joints['right_wrist'].y_norm - reachTop) / reachSpan) * 100).toFixed(0) + '%' : '-'
        };
        
        // ── Dive detection (velocity-based horizontal sweep) ──────────────
        if (joints['left_wrist'] && joints['right_wrist']) {
            const handCenterX = (joints['left_wrist'].x_norm + joints['right_wrist'].x_norm) / 2.0;
            const now = Date.now();
            this.gkHandHistory.push({ time: now, x: handCenterX });
            if (this.gkHandHistory.length > 5) this.gkHandHistory.shift();
            
            if (this.gkHandHistory.length >= 2) {
                const oldH = this.gkHandHistory[0];
                const newH = this.gkHandHistory[this.gkHandHistory.length - 1];
                const dt = (newH.time - oldH.time) / 1000.0;
                if (dt > 0) {
                    const velX = (newH.x - oldH.x) / dt;
                    this.handVelocityX = velX;
                    
                    const diveThreshold = 0.35;
                    if (velX > diveThreshold) {
                        this.gkDiveState = 'RIGHT_DIVE';
                        this.gkDiveTimer = now;
                    } else if (velX < -diveThreshold) {
                        this.gkDiveState = 'LEFT_DIVE';
                        this.gkDiveTimer = now;
                    }
                }
            }
        }
    }

    triggerAiShot() {
        // Randomly select one of the 9 target zones
        const zones = Object.keys(this.physics.zones3d);
        const targetZone = zones[Math.floor(Math.random() * zones.length)];
        
        let horiz = "CENTER";
        if (targetZone.endsWith("LEFT")) horiz = "LEFT";
        else if (targetZone.endsWith("RIGHT")) horiz = "RIGHT";
        
        let vert = "MID";
        if (targetZone.startsWith("TOP")) vert = "HIGH";
        else if (targetZone.startsWith("BOTTOM")) vert = "LOW";
        
        const diff = this.difficultyOptions[this.difficultyIndex];
        let power = 65;
        if (diff === "EASY") {
            power = 55 + Math.floor(Math.random() * 20); // 55-75%  (was MEDIUM)
        } else if (diff === "MEDIUM") {
            power = 75 + Math.floor(Math.random() * 20); // 75-95%  (was HARD)
        } else {
            power = 88 + Math.floor(Math.random() * 12); // 88-100% (new brutal HARD)
        }
        
        this.aiShotDirection = horiz;
        this.aiShotHeight = vert;
        this.aiShotPower = power;
        this.aiShotAngle = -15.0 + Math.random() * 30.0;
        
        this.ballTrajectory = this.physics.calculateTrajectory(
            targetZone,
            this.aiShotPower,
            this.aiShotAngle,
            true // isGoalkeeperMode — ball approaches from far goal post to camera
        );
        this.trajectoryIndex = 0;
        this.ballHistory = [];
        this.ballSpinAngle = 0;
        
        const promptEl = document.getElementById('game-prompt');
        if (promptEl) promptEl.classList.remove('active');
        
        this.gameState = "BALL_FLIGHT";
        this.audio.playSound('kick');
    }

    // STATE LOOP: GOALKEEPER MODE
    runKeeperLoop() {
        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Camera shake transform
        const shakeElapsed = Date.now() - this.cameraShakeTimer;
        const shakeDuration = 500;
        let shakeX = 0, shakeY = 0;
        if (this.cameraShakeTimer && shakeElapsed < shakeDuration) {
            const shakeStrength = (1.0 - shakeElapsed / shakeDuration) * 6;
            shakeX = (Math.random() - 0.5) * shakeStrength;
            shakeY = (Math.random() - 0.5) * shakeStrength;
        }
        ctx.save();
        ctx.translate(shakeX, shakeY);
        
        // 1. Draw stadium
        this.drawStadium();
        
        // Stadium lights slightly dim during kick preparation to build tension
        let elapsed = 0;
        if (this.gameState === "WAITING_SHOT") {
            elapsed = (Date.now() - this.gameStateTimer) / 1000.0;
            ctx.fillStyle = 'rgba(0, 0, 0, 0.22)';
            ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        } else if (this.gameState === "BALL_FLIGHT" || this.gameState === "RESULT_CELEBRATION") {
            elapsed = 3.5;
        }
        
        // 2. Draw penalty striker (silhouette)
        this.drawStriker(ctx, elapsed);
        
        // 3. Draw goal frame and net (rendered in front of kicker)
        this.drawGoal();
        
        // Draw the 5 goalkeeper save columns visually
        this.drawGKSaveZones();
        
        // If mouse fallback overrides glove controls
        if (!this.vision.webcamConnected) {
            this.leftGlovePos.x = this.mousePos.x - 65;
            this.leftGlovePos.y = this.mousePos.y;
            this.rightGlovePos.x = this.mousePos.x + 65;
            this.rightGlovePos.y = this.mousePos.y;
        }

        // Draw goalie skeletal helper overlay inside goal center (only in debug mode)
        if (this.debugMode) {
            this.drawGoalieSkeletalOverlay();
        }
        
        // Update components based on sub-state
        if (this.gameState === "GK_CALIBRATION") {
            // ── Inline GK Calibration Screen ────────────────────────────────
            const calibElapsed = (Date.now() - this.gkCalibPhaseTimer) / 1000.0;
            const calibDuration = 2.5; // seconds to hold position
            const remaining = Math.max(0, calibDuration - calibElapsed);
            
            // Dark overlay
            ctx.fillStyle = 'rgba(0, 0, 0, 0.72)';
            ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
            
            // Title
            ctx.fillStyle = '#00e5ff';
            ctx.font = 'bold 32px Outfit';
            ctx.textAlign = 'center';
            ctx.fillText('GOALKEEPER CALIBRATION', this.canvas.width / 2, 240);
            
            // Instruction
            ctx.fillStyle = '#ffffff';
            ctx.font = '20px Outfit';
            ctx.fillText('Stand naturally with hands in front of your chest', this.canvas.width / 2, 290);
            ctx.fillText('This sets your neutral hand position', this.canvas.width / 2, 318);
            
            // Animated hands icon
            const hx = this.canvas.width / 2;
            const hy = 390;
            const pulse = 0.85 + 0.15 * Math.sin(Date.now() * 0.006);
            ctx.save();
            ctx.translate(hx, hy);
            ctx.scale(pulse, pulse);
            ctx.fillStyle = 'rgba(0, 229, 255, 0.25)';
            ctx.strokeStyle = '#00e5ff';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.arc(-50, 0, 28, 0, 2 * Math.PI);
            ctx.fill();
            ctx.stroke();
            ctx.beginPath();
            ctx.arc(50, 0, 28, 0, 2 * Math.PI);
            ctx.fill();
            ctx.stroke();
            ctx.restore();
            
            // Progress bar
            const barW = 400;
            const barX = (this.canvas.width - barW) / 2;
            const barY = 460;
            const progress = Math.min(1.0, calibElapsed / calibDuration);
            
            // Background track
            ctx.fillStyle = 'rgba(255,255,255,0.15)';
            ctx.beginPath();
            if (typeof ctx.roundRect === 'function') {
                ctx.roundRect(barX, barY, barW, 14, 7);
            } else {
                ctx.rect(barX, barY, barW, 14);
            }
            ctx.fill();
            
            // Progress fill
            ctx.fillStyle = progress > 0.66 ? '#64ff64' : '#00e5ff';
            ctx.beginPath();
            if (typeof ctx.roundRect === 'function') {
                ctx.roundRect(barX, barY, barW * progress, 14, 7);
            } else {
                ctx.rect(barX, barY, barW * progress, 14);
            }
            ctx.fill();
            
            // Countdown text
            ctx.fillStyle = remaining < 0.5 ? '#64ff64' : '#ffffff';
            ctx.font = 'bold 22px Share Tech Mono';
            ctx.fillText(remaining > 0.05 ? `Calibrating... ${remaining.toFixed(1)}s` : 'CALIBRATED!', this.canvas.width / 2, 510);
            
            ctx.textAlign = 'left'; // reset
            
            // After 2.5s, finalize calibration
            if (calibElapsed >= calibDuration) {
                this.completeGkCalibration();
                this.gameState = "WAITING_SHOT";
                this.gameStateTimer = Date.now();
                document.getElementById('game-prompt').innerText = "AI PREPARING SHOT... GET READY!";
                document.getElementById('game-prompt').classList.add('active');
            }
            return; // Don't draw gloves/ball during calibration screen
        } else if (this.gameState === "WAITING_SHOT") {
            const cd = Math.max(1, Math.ceil(3.0 - elapsed));
            document.getElementById('game-prompt').innerText = `AI PREPARING SHOT... GET READY IN ${cd}...`;
            
            if (elapsed > 3.0) {
                this.triggerAiShot();
            }
        } else if (this.gameState === "BALL_FLIGHT") {
            if (this.trajectoryIndex < this.ballTrajectory.length) {
                const pt = this.ballTrajectory[this.trajectoryIndex];
                this.ballScreenPos = { x: pt.x, y: pt.y };
                this.ballRadius = pt.radius;
                this.ballPos3d = [pt.x_3d, pt.y_3d, pt.z];
                
                // Track ball history trail
                this.ballHistory = this.ballHistory || [];
                this.ballHistory.push({ x: pt.x, y: pt.y, r: pt.radius });
                if (this.ballHistory.length > 8) this.ballHistory.shift();
                
                // Spin animation
                this.ballSpinAngle = (this.ballSpinAngle || 0) + 0.2;
                
                // Check for save during the final section of flight (near goal line z >= 0.85)
                if (pt.z >= 0.85) {
                    this.checkForGkSave();
                }
                
                this.trajectoryIndex++;
            } else {
                // Trajectory finished, check final outcome if not already saved
                if (!this.outcomeText.includes("SAVE")) {
                    this.evaluateGkFinalOutcome();
                }
                this.gameState = "RESULT_CELEBRATION";
                this.gameStateTimer = Date.now();
                this.hasPlayedOutcomeSound = false;
            }
        } else if (this.gameState === "RESULT_CELEBRATION") {
            if (!this.hasPlayedOutcomeSound) {
                if (this.outcomeText.includes("SAVE")) {
                    this.audio.playSound('save');
                } else if (this.outcomeText.includes("GOAL")) {
                    this.audio.playSound('goal');
                } else {
                    this.audio.playSound('miss');
                }
                this.hasPlayedOutcomeSound = true;
                
                // Show announcement
                const outcomeEl = document.getElementById('game-outcome');
                outcomeEl.innerText = this.outcomeText;
                outcomeEl.className = "game-announcement outcome-announcement active " + 
                    (this.outcomeText.includes("SAVE") ? "text-green" : (this.outcomeText.includes("GOAL") ? "text-red" : "text-blue"));
                
                const promptEl = document.getElementById('game-prompt');
                promptEl.innerText = `AI Shot: ${this.aiShotDirection} | Height: ${this.aiShotHeight} | Power: ${this.aiShotPower}%`;
                promptEl.classList.add('active');
            }
            
            if (Date.now() - this.gameStateTimer > 2500) {
                const isSave = this.outcomeText.includes("SAVE");
                this.attempts++;
                if (isSave) this.saves++;
                
                // Clear outcomes & trigger direct transition
                document.getElementById('game-outcome').classList.remove('active');
                document.getElementById('game-prompt').classList.remove('active');
                this.completeAttempt();
            }
        }
        
        // 4. Draw Gloves
        this.drawGloves();
        
        // 5. Draw Ball
        if (this.gameState === "BALL_FLIGHT" || this.gameState === "RESULT_CELEBRATION" || this.gameState === "WAITING_SHOT") {
            if (this.gameState === "WAITING_SHOT") {
                this.ballScreenPos = { x: 512, y: 291 };
                this.ballRadius = 8;
            }
            this.drawBall();
        }
        
        // End camera shake transform scope
        ctx.restore();
        
        // Render corners camera preview
        this.drawCornersWebcam();
        
        // Update Live Broadcast elements & debug panels
        this.updateBroadcastOverlay();
        this.updateDebugPanel();
        
        // Update top HUD bar values
        this.updateHUDBar();
    }

    updateHUDBar() {
        const stat1Label = document.getElementById('hud-stat1-label');
        const stat1Val = document.getElementById('hud-stat1-val');
        const stat2Label = document.getElementById('hud-stat2-label');
        const stat2Val = document.getElementById('hud-stat2-val');
        const roundText = document.getElementById('hud-round-text');
        const trackingVal = document.getElementById('hud-tracking-val');
        const fpsVal = document.getElementById('hud-fps-val');
        
        if (this.gameModeType === "STRIKER") {
            if (stat1Label) stat1Label.innerText = "GOALS:";
            if (stat1Val) stat1Val.innerText = this.goals;
            if (stat2Label) stat2Label.innerText = "MISSES:";
            if (stat2Val) stat2Val.innerText = this.attempts - this.goals;
        } else {
            if (stat1Label) stat1Label.innerText = "SAVES:";
            if (stat1Val) stat1Val.innerText = this.saves;
            if (stat2Label) stat2Label.innerText = "GOALS:";
            if (stat2Val) stat2Val.innerText = this.attempts - this.saves;
        }
        
        if (roundText) {
            const currentRound = Math.min(this.maxAttempts, this.attempts + 1);
            roundText.innerText = `ROUND ${currentRound} / ${this.maxAttempts}`;
        }
        
        const status = this.vision.getTrackingStatus();
        if (trackingVal) {
            trackingVal.innerText = status;
            trackingVal.className = ""; // clear old classes
            if (status === 'GOOD') trackingVal.classList.add('status-green');
            else if (status === 'PARTIAL') trackingVal.classList.add('status-yellow');
            else trackingVal.classList.add('status-red');
        }
        
        if (fpsVal) {
            fpsVal.innerText = this.fps;
        }
    }

    drawGKSaveZones() {
        const ctx = this.ctx;
        const goalW = this.physics.gkGoalWidth;
        const goalH = this.physics.gkGoalHeight;
        const gt = this.physics.gkGoalTop;
        const gb = this.physics.gkGoalBottom;
        
        const columns = this.physics.gkColumns3d;
        
        // Determine player's active save zone
        let activeCol = 'CENTER';
        if (this.vision.smoothedJoints['left_wrist'] && this.vision.smoothedJoints['right_wrist']) {
            const lh = this.vision.smoothedJoints['left_hip'];
            const rh = this.vision.smoothedJoints['right_hip'];
            if (lh && rh) {
                const bodyCenterX = (lh.x_norm + rh.x_norm) / 2.0;
                const handCenterX = (this.vision.smoothedJoints['left_wrist'].x_norm + this.vision.smoothedJoints['right_wrist'].x_norm) / 2.0;
                const offset = handCenterX - bodyCenterX;
                const armLength = this.calibrationData.armLength || 0.45;
                const offsetRatio = offset / armLength;
                activeCol = this.physics.getGKColumnFromOffset(offsetRatio);
            }
        }
        
        // Apply horizontal dive override
        if (this.gkDiveState === 'LEFT_DIVE' && (Date.now() - this.gkDiveTimer < 600)) {
            activeCol = 'LEFT_DIVE';
        } else if (this.gkDiveState === 'RIGHT_DIVE' && (Date.now() - this.gkDiveTimer < 600)) {
            activeCol = 'RIGHT_DIVE';
        }
        
        const labelMap = {
            'LEFT_DIVE': 'FAR LEFT',
            'LEFT': 'LEFT',
            'CENTER': 'CENTER',
            'RIGHT': 'RIGHT',
            'RIGHT_DIVE': 'FAR RIGHT'
        };
        
        ctx.save();
        columns.forEach(col => {
            const xStart = this.physics.goalCenterX + (col.range[0] * goalW);
            const xEnd = this.physics.goalCenterX + (col.range[1] * goalW);
            const w = xEnd - xStart;
            
            const isActive = (col.id === activeCol);
            
            if (this.debugMode) {
                if (isActive) {
                    ctx.fillStyle = 'rgba(0, 255, 200, 0.25)';
                    ctx.strokeStyle = 'rgba(0, 255, 200, 0.9)';
                } else {
                    ctx.fillStyle = 'rgba(255, 255, 255, 0.05)';
                    ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
                }
                ctx.lineWidth = isActive ? 3 : 1;
                ctx.fillRect(xStart, gt, w, goalH);
                ctx.strokeRect(xStart, gt, w, goalH);
                
                ctx.fillStyle = isActive ? '#00ffc8' : 'rgba(255, 255, 255, 0.4)';
                ctx.font = 'bold 12px Share Tech Mono';
                ctx.textAlign = 'center';
                ctx.fillText(labelMap[col.id] || col.id, xStart + w / 2, gt - 8);
            } else {
                if (isActive) {
                    ctx.fillStyle = 'rgba(0, 255, 200, 0.025)';
                    ctx.strokeStyle = 'rgba(0, 255, 200, 0.1)';
                    ctx.lineWidth = 2;
                    ctx.fillRect(xStart, gt, w, goalH);
                    ctx.strokeRect(xStart, gt, w, goalH);
                    
                    ctx.fillStyle = 'rgba(0, 255, 200, 0.2)';
                    ctx.font = 'bold 12px Share Tech Mono';
                    ctx.textAlign = 'center';
                    ctx.fillText(labelMap[col.id] || col.id, xStart + w / 2, gt - 8);
                } else {
                    ctx.fillStyle = 'rgba(255, 255, 255, 0.005)';
                    ctx.strokeStyle = 'rgba(255, 255, 255, 0.02)';
                    ctx.lineWidth = 1;
                    ctx.fillRect(xStart, gt, w, goalH);
                    ctx.strokeRect(xStart, gt, w, goalH);
                }
            }
        });
        ctx.restore();
    }

    drawStriker(ctx, elapsed) {
        let phase = "STAND";
        let kickerX = 512 - 25;
        let kickerY = 291;
        let limbSwing = 0;
        
        if (this.gameModeType !== "GOALKEEPER") return;
        
        if (this.gameState === "WAITING_SHOT") {
            if (elapsed < 1.5) {
                phase = "STAND";
                kickerX = 512 - 25;
                kickerY = 291;
            } else if (elapsed < 3.0) {
                phase = "RUN";
                const runProgress = (elapsed - 1.5) / 1.5;
                kickerX = (512 - 25) + runProgress * 20;
                kickerY = 291;
                limbSwing = Math.sin(runProgress * 25) * 12; // Swing arms & legs
            } else {
                phase = "KICK";
                kickerX = 512 - 5;
                kickerY = 291;
            }
        } else if (this.gameState === "BALL_FLIGHT" || this.gameState === "RESULT_CELEBRATION") {
            phase = "POST_KICK";
            kickerX = 512 + 5; // stood a bit past the ball
            kickerY = 291;
        } else {
            return;
        }
        
        ctx.save();
        ctx.translate(kickerX, kickerY);
        
        // Scale the kicker (they are far away, so scale is small)
        const scale = 0.55;
        ctx.scale(scale, scale);
        
        // Ground Shadow
        ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
        ctx.beginPath();
        ctx.ellipse(0, 5, 25, 8, 0, 0, Math.PI * 2);
        ctx.fill();
        
        // Glow effect
        ctx.shadowBlur = 8;
        ctx.shadowColor = 'rgba(250, 100, 50, 0.4)';
        
        // Silhouette colors
        ctx.fillStyle = '#0d0d13';
        ctx.strokeStyle = '#f06432'; // neon orange outline
        ctx.lineWidth = 2.5;
        
        // Head
        ctx.beginPath();
        ctx.arc(0, -65, 10, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
        
        // Torso
        ctx.beginPath();
        ctx.moveTo(-6, -55);
        ctx.lineTo(6, -55);
        ctx.lineTo(8, -20);
        ctx.lineTo(-8, -20);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
        
        // Limbs based on phase
        ctx.lineWidth = 3.5;
        ctx.lineCap = 'round';
        
        if (phase === "STAND") {
            // Arms hanging down
            ctx.beginPath();
            ctx.moveTo(-7, -50);
            ctx.lineTo(-12, -25);
            ctx.moveTo(7, -50);
            ctx.lineTo(12, -25);
            ctx.stroke();
            
            // Legs standing slightly apart
            ctx.beginPath();
            ctx.moveTo(-5, -20);
            ctx.lineTo(-8, 10);
            ctx.moveTo(5, -20);
            ctx.lineTo(8, 10);
            ctx.stroke();
        } else if (phase === "RUN") {
            // Left arm forward, right arm back
            ctx.beginPath();
            ctx.moveTo(-7, -50);
            ctx.lineTo(-15 + limbSwing * 0.5, -30 + limbSwing * 0.3);
            ctx.moveTo(7, -50);
            ctx.lineTo(15 - limbSwing * 0.5, -30 - limbSwing * 0.3);
            ctx.stroke();
            
            // Left leg back, right leg forward
            ctx.beginPath();
            ctx.moveTo(-5, -20);
            ctx.lineTo(-10 + limbSwing, 5 - Math.abs(limbSwing) * 0.4);
            ctx.moveTo(5, -20);
            ctx.lineTo(10 - limbSwing, 5 - Math.abs(limbSwing) * 0.4);
            ctx.stroke();
        } else if (phase === "KICK") {
            // Right arm high for balance
            ctx.beginPath();
            ctx.moveTo(-7, -50);
            ctx.lineTo(-20, -40);
            ctx.moveTo(7, -50);
            ctx.lineTo(22, -65);
            ctx.stroke();
            
            // Left leg planting, right leg swinging high forward
            ctx.beginPath();
            ctx.moveTo(-5, -20);
            ctx.lineTo(-8, 10); // Planted leg
            ctx.moveTo(5, -20);
            ctx.lineTo(25, -5); // Kicking leg
            ctx.stroke();
        } else if (phase === "POST_KICK") {
            // Post kick balance pose
            ctx.beginPath();
            ctx.moveTo(-7, -50);
            ctx.lineTo(-15, -35);
            ctx.moveTo(7, -50);
            ctx.lineTo(15, -45);
            ctx.stroke();
            
            // Standing on right leg, left leg trailing
            ctx.beginPath();
            ctx.moveTo(-5, -20);
            ctx.lineTo(-15, 5); // trailing leg
            ctx.moveTo(5, -20);
            ctx.lineTo(2, 10); // standing leg
            ctx.stroke();
        }
        
        ctx.restore();
    }

    checkForGkSave() {
        if (this.outcomeText.includes("SAVE")) return; // Already registered
        
        const bx = this.ballScreenPos.x;
        const by = this.ballScreenPos.y;
        
        const ballX3D = this.ballPos3d[0];
        const ballCol = this.physics.getGKColumn(ballX3D);
        
        // Active column detection
        let activeCol = 'CENTER';
        if (this.vision.smoothedJoints['left_wrist'] && this.vision.smoothedJoints['right_wrist']) {
            const lh = this.vision.smoothedJoints['left_hip'];
            const rh = this.vision.smoothedJoints['right_hip'];
            if (lh && rh) {
                const bodyCenterX = (lh.x_norm + rh.x_norm) / 2.0;
                const handCenterX = (this.vision.smoothedJoints['left_wrist'].x_norm + this.vision.smoothedJoints['right_wrist'].x_norm) / 2.0;
                const offset = handCenterX - bodyCenterX;
                const armLength = this.calibrationData.armLength || 0.45;
                const offsetRatio = offset / armLength;
                activeCol = this.physics.getGKColumnFromOffset(offsetRatio);
            }
        }
        
        // Dive override
        if (this.gkDiveState === 'LEFT_DIVE' && (Date.now() - this.gkDiveTimer < 600)) {
            activeCol = 'LEFT_DIVE';
        } else if (this.gkDiveState === 'RIGHT_DIVE' && (Date.now() - this.gkDiveTimer < 600)) {
            activeCol = 'RIGHT_DIVE';
        }
        
        let saved = false;
        if (ballCol === activeCol) {
            saved = true;
        } else {
            // Screen-space overlap fallback (direct touch)
            const dl = Math.sqrt(Math.pow(bx - this.leftGlovePos.x, 2) + Math.pow(by - this.leftGlovePos.y, 2));
            const dr = Math.sqrt(Math.pow(bx - this.rightGlovePos.x, 2) + Math.pow(by - this.rightGlovePos.y, 2));
            const limit = this.ballRadius + this.gloveRadius;
            if (dl < limit || dr < limit) {
                saved = true;
            }
        }
        
        if (saved) {
            this.outcomeText = "GREAT SAVE!!!";
            this.outcomeColor = "#64ff64";
            
            // Camera shake on save
            this.cameraShakeTimer = Date.now();
            
            // Build deflection frames: ball bounces back towards center
            const deflectionFrames = [];
            const cx = this.ballScreenPos.x;
            const cy = this.ballScreenPos.y;
            const cr = this.ballRadius;
            const numDefl = 18;
            
            for (let i = 1; i <= numDefl; i++) {
                const t = i / numDefl;
                const deflX = cx + (ballX3D < 0 ? -1 : 1) * t * 80; // Bounce sideways
                const deflY = cy + t * 60; // Fall downward
                const deflR = Math.max(4, cr * (1.0 - t * 0.5));
                deflectionFrames.push({
                    x: deflX,
                    y: deflY,
                    z: 1.0 - t * 0.3,
                    x_3d: ballX3D,
                    y_3d: 0.0,
                    radius: deflR
                });
            }
            
            // Replace remaining trajectory with deflection
            this.ballTrajectory = deflectionFrames;
            this.trajectoryIndex = 0;
        }
    }

    evaluateGkFinalOutcome() {
        const ballX = this.ballScreenPos.x;
        const ballY = this.ballScreenPos.y;
        
        const gl = this.physics.gkGoalLeft;
        const gr = this.physics.gkGoalRight;
        const gt = this.physics.gkGoalTop;
        const gb = this.physics.gkGoalBottom;
        
        const inside = (ballX >= gl && ballX <= gr && ballY >= gt && ballY <= gb);
        if (inside) {
            this.outcomeText = "GOAL CONCEDED!";
            this.outcomeColor = "#ff6464";
            // Screen flash on goal
            this.screenFlashTimer = Date.now();
        } else {
            this.outcomeText = "SHOT WIDE / MISSED!";
            this.outcomeColor = "#a0a0c0";
        }
    }

    completeAttempt() {
        if (this.attempts >= this.maxAttempts) {
            // Game finished, check leaderboard
            const finalScore = this.gameModeType === "STRIKER" ? this.goals : this.saves;
            this.qualifyingScore = finalScore;
            this.leaderboardMode = this.gameModeType;
            
            if (this.qualifiesForLeaderboard(finalScore, this.leaderboardMode)) {
                const modeStr = this.gameModeType === "STRIKER" ? "Goals" : "Saves";
                const diff = this.difficultyOptions[this.difficultyIndex];
                document.getElementById('score-msg').innerText = `You scored ${finalScore} ${modeStr} on ${diff}!`;
                this.showView("name-input-view");
                this.state = "NAME_INPUT";
                document.getElementById('player-name-input').focus();
            } else {
                this.renderLeaderboardTable();
                this.showView("leaderboard-view");
                this.state = "LEADERBOARD";
            }
        } else {
            this.resetAttempt();
            if (this.gameModeType === "STRIKER") {
                this.state = "PLAY_STRIKER";
            } else {
                this.state = "PLAY_GOALKEEPER";
            }
        }
    }

    // LEADERBOARD LOCAL STORAGE LOGIC
    qualifiesForLeaderboard(score, mode) {
        const scores = this.getScores();
        const modeScores = scores.filter(s => s.mode === mode);
        if (modeScores.length < 10) return true;
        
        // Is higher than the lowest highscore in that mode
        modeScores.sort((a, b) => b.score - a.score);
        return score > modeScores[modeScores.length - 1].score;
    }

    getScores() {
        const data = localStorage.getItem('vision_football_scores');
        return data ? JSON.parse(data) : [];
    }

    addLeaderboardEntry(name) {
        const scores = this.getScores();
        const diff = this.difficultyOptions[this.difficultyIndex];
        
        scores.push({
            name: name,
            mode: this.leaderboardMode,
            score: this.qualifyingScore,
            difficulty: diff,
            date: new Date().toISOString().slice(0, 10)
        });
        
        // Sort
        scores.sort((a, b) => b.score - a.score);
        
        // Limit
        localStorage.setItem('vision_football_scores', JSON.stringify(scores.slice(0, 50)));
    }

    renderLeaderboardTable() {
        const scores = this.getScores().slice(0, 10); // top 10
        const tbody = document.getElementById('leaderboard-tbody');
        tbody.innerHTML = "";
        
        scores.forEach((s, idx) => {
            const tr = document.createElement('tr');
            tr.className = `rank-${idx + 1}`;
            
            const rk = idx + 1;
            const rankLabel = rk === 1 ? "🥇 01" : (rk === 2 ? "🥈 02" : (rk === 3 ? "🥉 03" : `${rk < 10 ? '0' : ''}${rk}`));
            const modeLabel = s.mode === "STRIKER" ? "STRIKER" : "GK MODE";
            const scoreLabel = s.mode === "STRIKER" ? `${s.score} Goals` : `${s.score} Saves`;
            
            tr.innerHTML = `
                <td>${rankLabel}</td>
                <td>${s.name}</td>
                <td>${modeLabel}</td>
                <td>${scoreLabel}</td>
                <td>${s.difficulty}</td>
            `;
            tbody.appendChild(tr);
        });
    }

    drawStrikerScene(ctx) {
        const w = this.canvas.width;
        const h = this.canvas.height;
        const skyH = Math.floor(h * 0.45);
        
        // Sky Sunset Gradient (Deep Purple to vibrant Orange/Magenta)
        const skyGrad = ctx.createLinearGradient(0, 0, 0, skyH);
        skyGrad.addColorStop(0, '#100b26');
        skyGrad.addColorStop(0.4, '#2c1654');
        skyGrad.addColorStop(0.8, '#881b5c');
        skyGrad.addColorStop(1, '#f35a38');
        ctx.fillStyle = skyGrad;
        ctx.fillRect(0, 0, w, skyH);
        
        // Draw Stadium Stands/Crowd Silhouette at the horizon
        ctx.fillStyle = '#100820';
        ctx.beginPath();
        ctx.moveTo(0, skyH);
        for (let x = 0; x <= w; x += 40) {
            const standY = skyH - 15 - Math.sin(x * 0.05) * 8 - (x % 120 === 0 ? 12 : 0);
            ctx.lineTo(x, standY);
        }
        ctx.lineTo(w, skyH);
        ctx.closePath();
        ctx.fill();
        
        // Crowd light sparkles
        ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
        for (let i = 0; i < 45; i++) {
            const rx = Math.random() * w;
            const ry = skyH - 5 - Math.random() * 20;
            ctx.fillRect(rx, ry, 2, 2);
        }
        
        // Floodlights
        const floodlights = [w * 0.12, w * 0.38, w * 0.62, w * 0.88];
        floodlights.forEach(fx => {
            // Draw floodlight beams
            const beamGrad = ctx.createRadialGradient(fx, 60, 5, fx, 180, 240);
            beamGrad.addColorStop(0, 'rgba(255, 255, 220, 0.35)');
            beamGrad.addColorStop(0.4, 'rgba(255, 240, 180, 0.1)');
            beamGrad.addColorStop(1, 'rgba(0, 0, 0, 0)');
            
            ctx.fillStyle = beamGrad;
            ctx.beginPath();
            ctx.moveTo(fx - 40, 60);
            ctx.lineTo(fx + 40, 60);
            ctx.lineTo(fx + 220, h);
            ctx.lineTo(fx - 220, h);
            ctx.closePath();
            ctx.fill();
            
            // Light bulbs glow
            ctx.save();
            ctx.shadowBlur = 35;
            ctx.shadowColor = '#fff6d0';
            ctx.fillStyle = '#ffffff';
            ctx.beginPath();
            ctx.arc(fx, 60, 15, 0, 2 * Math.PI);
            ctx.fill();
            ctx.restore();
        });
        
        // Immersive Pitch Gradient
        const pitchH = h - skyH;
        const pitchGrad = ctx.createLinearGradient(0, skyH, 0, h);
        pitchGrad.addColorStop(0, '#0c4912');
        pitchGrad.addColorStop(0.5, '#136e1c');
        pitchGrad.addColorStop(1, '#1e9628');
        ctx.fillStyle = pitchGrad;
        ctx.fillRect(0, skyH, w, pitchH);
        
        // 3D Pitch Striping converging to horizon
        ctx.fillStyle = 'rgba(255, 255, 255, 0.04)';
        for (let i = 0; i < 12; i++) {
            const yStart = skyH + (i * pitchH / 12);
            const yEnd = skyH + ((i + 1) * pitchH / 12);
            if (i % 2 === 0) {
                ctx.beginPath();
                const xLeftTop = (yStart - skyH) / pitchH * -120;
                const xRightTop = w + (yStart - skyH) / pitchH * 120;
                const xLeftBot = (yEnd - skyH) / pitchH * -120;
                const xRightBot = w + (yEnd - skyH) / pitchH * 120;
                
                ctx.moveTo(xLeftTop, yStart);
                ctx.lineTo(xRightTop, yStart);
                ctx.lineTo(xRightBot, yEnd);
                ctx.lineTo(xLeftBot, yEnd);
                ctx.closePath();
                ctx.fill();
            }
        }
        
        // 3D Lines (touchlines & penalty arc)
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.28)';
        ctx.lineWidth = 3;
        
        // Ground touchlines converging to goal posts
        ctx.beginPath();
        ctx.moveTo(-150, h);
        ctx.lineTo(this.physics.strikerGoalLeft, this.physics.strikerGoalBottom);
        ctx.moveTo(w + 150, h);
        ctx.lineTo(this.physics.strikerGoalRight, this.physics.strikerGoalBottom);
        ctx.stroke();
        
        // Goal line
        ctx.beginPath();
        ctx.moveTo(this.physics.strikerGoalLeft, this.physics.strikerGoalBottom);
        ctx.lineTo(this.physics.strikerGoalRight, this.physics.strikerGoalBottom);
        ctx.stroke();
        
        // Penalty Area Box lines
        ctx.beginPath();
        const pBoxLeftFar = this.physics.strikerGoalLeft - 60;
        const pBoxRightFar = this.physics.strikerGoalRight + 60;
        const pBoxLeftNear = -50;
        const pBoxRightNear = w + 50;
        const pBoxTop = this.physics.strikerGoalBottom;
        const pBoxBottom = Math.floor(h * 0.76);
        
        ctx.moveTo(pBoxLeftFar, pBoxTop);
        ctx.lineTo(pBoxRightFar, pBoxTop);
        ctx.moveTo(pBoxLeftFar, pBoxTop);
        ctx.lineTo(pBoxLeftNear, pBoxBottom);
        ctx.moveTo(pBoxRightFar, pBoxTop);
        ctx.lineTo(pBoxRightNear, pBoxBottom);
        ctx.moveTo(pBoxLeftNear, pBoxBottom);
        ctx.lineTo(pBoxRightNear, pBoxBottom);
        ctx.stroke();
        
        // Penalty Spot (3D circle)
        ctx.fillStyle = 'rgba(255, 255, 255, 0.85)';
        ctx.shadowBlur = 10;
        ctx.shadowColor = '#ffffff';
        ctx.beginPath();
        ctx.ellipse(w / 2, this.physics.penaltySpot.y + 4, 16, 6, 0, 0, 2 * Math.PI);
        ctx.fill();
        ctx.shadowBlur = 0;
    }

    drawFirstPersonGoal(ctx) {
        const gl = this.physics.strikerGoalLeft;
        const gr = this.physics.strikerGoalRight;
        const gt = this.physics.strikerGoalTop;
        const gb = this.physics.strikerGoalBottom;
        const gw = gr - gl;
        const gh = gb - gt;
        
        // Draw net backing depth box (sides & top extending back)
        ctx.fillStyle = 'rgba(10, 40, 15, 0.55)';
        ctx.beginPath();
        ctx.rect(gl, gt, gw, gh);
        ctx.fill();
        
        // Goal net mesh drawing with optional ripple distortion
        const flashElapsed = this.screenFlashTimer ? Date.now() - this.screenFlashTimer : 9999;
        const rippleActive = (flashElapsed < 800) && this.outcomeText.includes("GOAL");
        
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.22)';
        ctx.lineWidth = 1.5;
        const spacing = 18;
        
        for (let x = gl + spacing; x < gr; x += spacing) {
            ctx.beginPath();
            ctx.moveTo(x, gt);
            for (let y = gt; y <= gb; y += 10) {
                let ripple = 0;
                if (rippleActive) {
                    const dx = Math.abs(x - this.ballScreenPos.x);
                    const dy = Math.abs(y - this.ballScreenPos.y);
                    const dist = Math.sqrt(dx*dx + dy*dy);
                    ripple = Math.max(0, 32 - dist * 0.15) * Math.sin(y * 0.12 - flashElapsed * 0.02) * (1.0 - flashElapsed / 800);
                }
                ctx.lineTo(x + ripple, y);
            }
            ctx.stroke();
        }
        
        for (let y = gt + spacing; y < gb; y += spacing) {
            ctx.beginPath();
            ctx.moveTo(gl, y);
            for (let x = gl; x <= gr; x += 10) {
                let ripple = 0;
                if (rippleActive) {
                    const dx = Math.abs(x - this.ballScreenPos.x);
                    const dy = Math.abs(y - this.ballScreenPos.y);
                    const dist = Math.sqrt(dx*dx + dy*dy);
                    ripple = Math.max(0, 32 - dist * 0.15) * Math.sin(x * 0.12 - flashElapsed * 0.02) * (1.0 - flashElapsed / 800);
                }
                ctx.lineTo(x, y + ripple);
            }
            ctx.stroke();
        }
        
        // Goalposts & Crossbar (glossy 3D look)
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 14;
        ctx.lineCap = 'square';
        ctx.lineJoin = 'miter';
        
        ctx.shadowBlur = 15;
        ctx.shadowColor = 'rgba(255, 255, 255, 0.45)';
        ctx.beginPath();
        ctx.moveTo(gl, gb);
        ctx.lineTo(gl, gt);
        ctx.lineTo(gr, gt);
        ctx.lineTo(gr, gb);
        ctx.stroke();
        ctx.shadowBlur = 0;
        
        // Inner post shadows for depth
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.15)';
        ctx.lineWidth = 4;
        ctx.beginPath();
        ctx.moveTo(gl + 7, gb);
        ctx.lineTo(gl + 7, gt + 7);
        ctx.lineTo(gr - 7, gt + 7);
        ctx.lineTo(gr - 7, gb);
        ctx.stroke();
    }

    updateGoalkeeperAnimation(deltaTime) {
        const anim = this.keeperAnim;
        anim.elapsed += deltaTime;

        // Sync with AI state changes
        const logicState = this.keeperAiState;
        
        if (logicState === "READING") {
            if (anim.state !== 0) {
                anim.state = 0; // IDLE
                anim.elapsed = 0;
            }
        } else if (logicState === "ANTICIPATING") {
            if (anim.state !== 1 && anim.state !== 2) {
                anim.state = this.keeperDiveDir === "LEFT" ? 1 : 2; // ANTICIPATE
                anim.elapsed = 0;
            }
        } else if (logicState === "DIVING") {
            if (anim.state === 1 || anim.state === 2) {
                anim.state = this.keeperDiveDir === "LEFT" ? 3 : 4; // PUSH
                anim.elapsed = 0;
            }
            // Transition from PUSH to DIVE after 150ms
            if ((anim.state === 3 || anim.state === 4) && anim.elapsed > 150) {
                anim.state = this.keeperDiveDir === "LEFT" ? 5 : 6; // DIVE
                anim.elapsed = 0;
            }
        } else if (logicState === "LANDED") {
            if (anim.state !== 9 && anim.state !== 10 && anim.state !== 11) {
                anim.state = this.keeperDiveDir === "LEFT" ? 9 : 10; // LAND
                anim.elapsed = 0;
            }
            if ((anim.state === 9 || anim.state === 10) && anim.elapsed > 300) {
                anim.state = 11; // RECOVER
                anim.elapsed = 0;
            }
        }

        // --- Calculate visual variables based on anim.state and elapsed ---
        
        // Base logical position
        const targetX = this.keeperX;
        const targetY = this.keeperY;
        
        // Interpolate visual position (visX, visY) towards logical target
        // Different states have different speeds/easings
        let smoothing = 0.2; 
        
        let leanX = 0;
        let leanY = 0;
        
        anim.headRot = 0;
        anim.torsoRot = 0;
        anim.leftArmRot = 0;
        anim.rightArmRot = 0;
        anim.leftLegRot = 0;
        anim.rightLegRot = 0;
        anim.shadowScale = 1.0;
        anim.shadowWidth = 45;

        // Head tracking based on ball
        const ballX = this.ballScreenPos.x;
        const headOffset = (ballX - anim.visX) * 0.05;
        anim.headRot = Math.max(-25, Math.min(25, headOffset));

        switch(anim.state) {
            case 0: // IDLE
                // Subtle breathing and swaying
                leanY = Math.sin(anim.elapsed * 0.003) * 3;
                leanX = Math.cos(anim.elapsed * 0.001) * 2;
                anim.leftArmRot = 15 + Math.sin(anim.elapsed * 0.002) * 2;
                anim.rightArmRot = -15 - Math.sin(anim.elapsed * 0.002) * 2;
                anim.leftLegRot = 5;
                anim.rightLegRot = -5;
                smoothing = 0.1;
                break;
                
            case 1: // ANTICIPATE LEFT
            case 2: // ANTICIPATE RIGHT
                {
                    const dir = anim.state === 1 ? -1 : 1;
                    const progress = Math.min(1.0, anim.elapsed / 150.0);
                    leanY = 15 * progress; // crouch
                    leanX = 10 * dir * progress; // shift weight
                    anim.torsoRot = 10 * dir * progress;
                    anim.leftArmRot = 20 * progress;
                    anim.rightArmRot = -20 * progress;
                    // Bend legs
                    anim.leftLegRot = 10 * dir * progress;
                    anim.rightLegRot = 10 * dir * progress;
                    smoothing = 0.3;
                }
                break;

            case 3: // PUSH LEFT
            case 4: // PUSH RIGHT
                {
                    const dir = anim.state === 3 ? -1 : 1;
                    const progress = Math.min(1.0, anim.elapsed / 150.0);
                    leanY = 15 - (15 * progress); // spring up
                    anim.torsoRot = 20 * dir * progress;
                    
                    // Leading arm extends, trailing arm pulls back
                    if (dir === -1) {
                        anim.leftArmRot = 60 * progress;
                        anim.rightArmRot = -30 * progress;
                        anim.rightLegRot = -45 * progress; // push foot
                    } else {
                        anim.rightArmRot = -60 * progress;
                        anim.leftArmRot = 30 * progress;
                        anim.leftLegRot = 45 * progress; // push foot
                    }
                    smoothing = 0.4;
                }
                break;

            case 5: // DIVE LEFT
            case 6: // DIVE RIGHT
                {
                    const dir = anim.state === 5 ? -1 : 1;
                    // Fully committed dive pose
                    
                    // Differentiate High/Mid/Low dive
                    const hType = this.keeperDiveHeight;
                    let diveAngle = 75; // almost horizontal
                    if (hType === "HIGH") diveAngle = 45; // angle upwards
                    else if (hType === "LOW") diveAngle = 85; // very flat
                    
                    anim.torsoRot = diveAngle * dir;
                    
                    // Head stays aligned but looks towards ball
                    anim.headRot = (-diveAngle * dir * 0.4) + headOffset * 0.5;
                    
                    // Arms stretch
                    if (dir === -1) {
                        anim.leftArmRot = 95;  // fully extended towards ball
                        anim.rightArmRot = 40; // trails behind
                        anim.leftLegRot = 20;
                        anim.rightLegRot = 60; // trailing leg
                    } else {
                        anim.rightArmRot = -95;
                        anim.leftArmRot = -40;
                        anim.rightLegRot = -20;
                        anim.leftLegRot = -60;
                    }
                    
                    anim.shadowScale = hType === "HIGH" ? 0.6 : (hType === "LOW" ? 1.2 : 0.8);
                    anim.shadowWidth = 70; // stretch shadow during dive
                    
                    // Check for outcome collision reaction
                    if (this.outcomeText && this.outcomeText.includes("SAVE")) {
                        // Recoil glove slightly if save registered
                        if (dir === -1) anim.leftArmRot -= 15;
                        else anim.rightArmRot += 15;
                        
                        anim.torsoRot -= 5 * dir; // body recoil
                    }
                    
                    smoothing = 0.5;
                }
                break;

            case 9:  // LAND LEFT
            case 10: // LAND RIGHT
                {
                    const dir = anim.state === 9 ? -1 : 1;
                    const progress = Math.min(1.0, anim.elapsed / 300.0);
                    
                    // Transition from horizontal to slightly tucked ground position
                    anim.torsoRot = (75 * dir) * (1.0 - progress * 0.3);
                    anim.shadowScale = 1.0;
                    anim.shadowWidth = 60 - (15 * progress);
                    
                    if (dir === -1) {
                        anim.leftArmRot = 95 * (1.0 - progress * 0.5);
                        anim.rightArmRot = 40 * (1.0 - progress);
                    } else {
                        anim.rightArmRot = -95 * (1.0 - progress * 0.5);
                        anim.leftArmRot = -40 * (1.0 - progress);
                    }
                    smoothing = 0.3;
                }
                break;

            case 11: // RECOVER
                {
                    const progress = Math.min(1.0, anim.elapsed / 500.0);
                    anim.torsoRot = anim.torsoRot * (1.0 - progress);
                    anim.leftArmRot = anim.leftArmRot * (1.0 - progress);
                    anim.rightArmRot = anim.rightArmRot * (1.0 - progress);
                    anim.leftLegRot = anim.leftLegRot * (1.0 - progress);
                    anim.rightLegRot = anim.rightLegRot * (1.0 - progress);
                    smoothing = 0.15;
                }
                break;
        }

        // Apply interpolation to visual position (independent of collision)
        anim.visX += ((targetX + leanX) - anim.visX) * smoothing;
        anim.visY += ((targetY + leanY) - anim.visY) * smoothing;
    }

    drawFirstPersonGoalkeeper(ctx) {
        // Calculate deltaTime for smooth animation
        const now = Date.now();
        if (!this._lastGkTime) this._lastGkTime = now;
        const deltaTime = now - this._lastGkTime;
        this._lastGkTime = now;
        
        this.updateGoalkeeperAnimation(deltaTime);
        const anim = this.keeperAnim;
        
        const goalieHeight = Math.floor(this.physics.strikerGoalHeight * 0.32);
        
        ctx.save();
        ctx.translate(anim.visX, anim.visY);
        
        // --- 1. Dynamic Ground Shadow ---
        ctx.fillStyle = `rgba(0, 0, 0, ${0.4 * anim.shadowScale})`;
        ctx.beginPath();
        // Base shadow Y should anchor near the feet (which are roughly +goalieHeight*0.45 relative to center)
        ctx.ellipse(0, goalieHeight * 0.45, anim.shadowWidth * anim.shadowScale, 12 * anim.shadowScale, 0, 0, Math.PI * 2);
        ctx.fill();

        // --- 2. Goalkeeper Renderer ---
        
        // Colors
        const shirtColor = '#ccff00';     // Fluorescent yellow/green
        const shirtDark = '#99cc00';
        const shortsColor = '#1a1a24';    // Dark navy/black
        const skinColor = '#e5aa7a';      // Natural skin tone
        const gloveColor = '#ffffff';     // White pro gloves
        const bootColor = '#ff2a2a';      // Bright red boots
        const hairColor = '#2d1b11';
        
        // Helper to convert deg to rad
        const rad = (deg) => deg * Math.PI / 180;
        
        // Draw Torso Center Pivot
        ctx.rotate(rad(anim.torsoRot));
        
        // Draw Legs (behind torso)
        ctx.lineWidth = goalieHeight * 0.12;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        
        // Left Leg
        ctx.save();
        ctx.translate(-goalieHeight * 0.1, goalieHeight * 0.15);
        ctx.rotate(rad(anim.leftLegRot));
        // Thigh
        ctx.strokeStyle = shortsColor;
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(0, goalieHeight * 0.15);
        ctx.stroke();
        // Calf
        ctx.strokeStyle = skinColor;
        ctx.beginPath();
        ctx.moveTo(0, goalieHeight * 0.15);
        ctx.lineTo(0, goalieHeight * 0.35);
        ctx.stroke();
        // Boot
        ctx.fillStyle = bootColor;
        ctx.fillRect(-goalieHeight * 0.08, goalieHeight * 0.35, goalieHeight * 0.16, goalieHeight * 0.08);
        ctx.restore();

        // Right Leg
        ctx.save();
        ctx.translate(goalieHeight * 0.1, goalieHeight * 0.15);
        ctx.rotate(rad(anim.rightLegRot));
        ctx.strokeStyle = shortsColor;
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(0, goalieHeight * 0.15);
        ctx.stroke();
        ctx.strokeStyle = skinColor;
        ctx.beginPath();
        ctx.moveTo(0, goalieHeight * 0.15);
        ctx.lineTo(0, goalieHeight * 0.35);
        ctx.stroke();
        ctx.fillStyle = bootColor;
        ctx.fillRect(-goalieHeight * 0.08, goalieHeight * 0.35, goalieHeight * 0.16, goalieHeight * 0.08);
        ctx.restore();

        // Draw Torso/Jersey
        // Create 3D gradient for jersey
        const grad = ctx.createLinearGradient(-goalieHeight * 0.2, 0, goalieHeight * 0.2, 0);
        grad.addColorStop(0, shirtDark);
        grad.addColorStop(0.5, shirtColor);
        grad.addColorStop(1, shirtDark);
        
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.moveTo(-goalieHeight * 0.22, -goalieHeight * 0.25); // Top left shoulder
        ctx.lineTo(goalieHeight * 0.22, -goalieHeight * 0.25);  // Top right shoulder
        ctx.lineTo(goalieHeight * 0.18, goalieHeight * 0.2);    // Bottom right hip
        ctx.lineTo(-goalieHeight * 0.18, goalieHeight * 0.2);   // Bottom left hip
        ctx.closePath();
        ctx.fill();
        
        // Collar
        ctx.fillStyle = shortsColor;
        ctx.beginPath();
        ctx.arc(0, -goalieHeight * 0.25, goalieHeight * 0.08, 0, Math.PI);
        ctx.fill();
        
        // Pattern on Jersey
        ctx.strokeStyle = 'rgba(0,0,0,0.1)';
        ctx.lineWidth = 4;
        ctx.beginPath();
        ctx.moveTo(-goalieHeight * 0.1, -goalieHeight * 0.2);
        ctx.lineTo(goalieHeight * 0.1, goalieHeight * 0.1);
        ctx.stroke();
        
        // Head (Attached to torso but can rotate independently)
        ctx.save();
        ctx.translate(0, -goalieHeight * 0.25); // Move to neck
        ctx.rotate(rad(anim.headRot - anim.torsoRot)); // Counter-rotate torso to track ball
        
        // Neck
        ctx.fillStyle = skinColor;
        ctx.fillRect(-goalieHeight * 0.05, -goalieHeight * 0.06, goalieHeight * 0.1, goalieHeight * 0.06);
        
        // Face
        ctx.beginPath();
        ctx.arc(0, -goalieHeight * 0.18, goalieHeight * 0.14, 0, 2 * Math.PI);
        ctx.fill();
        
        // Hair
        ctx.fillStyle = hairColor;
        ctx.beginPath();
        ctx.arc(0, -goalieHeight * 0.22, goalieHeight * 0.14, Math.PI, 2 * Math.PI);
        ctx.fill();
        ctx.restore();

        // Arms (Procedural IK-style)
        ctx.lineWidth = goalieHeight * 0.11;
        
        // Left Arm
        ctx.save();
        ctx.translate(-goalieHeight * 0.2, -goalieHeight * 0.2);
        ctx.rotate(rad(anim.leftArmRot));
        // Upper Arm (Jersey)
        ctx.strokeStyle = shirtColor;
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(-goalieHeight * 0.2, goalieHeight * 0.15);
        ctx.stroke();
        // Lower Arm (Skin)
        ctx.strokeStyle = skinColor;
        ctx.beginPath();
        ctx.moveTo(-goalieHeight * 0.2, goalieHeight * 0.15);
        ctx.lineTo(-goalieHeight * 0.35, goalieHeight * 0.35);
        ctx.stroke();
        // Glove
        ctx.fillStyle = gloveColor;
        ctx.beginPath();
        ctx.arc(-goalieHeight * 0.38, goalieHeight * 0.38, goalieHeight * 0.1, 0, 2 * Math.PI);
        ctx.fill();
        // Glove details
        ctx.fillStyle = '#000';
        ctx.fillRect(-goalieHeight * 0.4, goalieHeight * 0.35, goalieHeight * 0.1, goalieHeight * 0.04);
        ctx.restore();

        // Right Arm
        ctx.save();
        ctx.translate(goalieHeight * 0.2, -goalieHeight * 0.2);
        ctx.rotate(rad(anim.rightArmRot));
        // Upper Arm
        ctx.strokeStyle = shirtColor;
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(goalieHeight * 0.2, goalieHeight * 0.15);
        ctx.stroke();
        // Lower Arm
        ctx.strokeStyle = skinColor;
        ctx.beginPath();
        ctx.moveTo(goalieHeight * 0.2, goalieHeight * 0.15);
        ctx.lineTo(goalieHeight * 0.35, goalieHeight * 0.35);
        ctx.stroke();
        // Glove
        ctx.fillStyle = gloveColor;
        ctx.beginPath();
        ctx.arc(goalieHeight * 0.38, goalieHeight * 0.38, goalieHeight * 0.1, 0, 2 * Math.PI);
        ctx.fill();
        ctx.fillStyle = '#000';
        ctx.fillRect(goalieHeight * 0.3, goalieHeight * 0.35, goalieHeight * 0.1, goalieHeight * 0.04);
        ctx.restore();
        
        ctx.restore(); // Undo translations

        // 3. Draw Debug Logical Hitboxes over the visual render
        if (this.debugMode || this.trainingMode) {
            const def = this._keeperColliderDef;
            if (def && def.length > 0) {
                ctx.save();
                // Render at true logical position (this.keeperX, this.keeperY)
                ctx.translate(this.keeperX, this.keeperY);
                if (this.keeperRotation !== 0) {
                    ctx.rotate((this.keeperRotation * Math.PI) / 180.0);
                }
                
                ctx.lineWidth = 2;
                for (const col of def) {
                    ctx.strokeStyle = '#ff00ff'; // neon magenta
                    ctx.fillStyle = 'rgba(255, 0, 255, 0.2)';
                    
                    // Rect is centered at (col.ox, col.oy) with size (col.w, col.h)
                    ctx.fillRect(col.ox - col.w / 2, col.oy - col.h / 2, col.w, col.h);
                    ctx.strokeRect(col.ox - col.w / 2, col.oy - col.h / 2, col.w, col.h);
                    
                    // Text label inside hitbox
                    ctx.fillStyle = '#ffffff';
                    ctx.font = 'bold 9px Share Tech Mono';
                    ctx.textAlign = 'center';
                    ctx.fillText(col.id.toUpperCase(), col.ox, col.oy + 3);
                }
                ctx.restore();
            }
        }
    }

    drawStrikerBallAtRest(ctx) {
        const x = this.physics.penaltySpot.x;
        const y = this.physics.penaltySpot.y;
        const r = this.physics.startRadius;
        
        const pulse = 1.0 + Math.sin(Date.now() * 0.005) * 0.08;
        
        ctx.save();
        ctx.shadowBlur = 20 + Math.sin(Date.now() * 0.005) * 8;
        ctx.shadowColor = 'rgba(255, 230, 80, 0.7)';
        
        // Shadow under ball
        const shadowGrad = ctx.createRadialGradient(x, y + r - 5, 2, x, y + r + 5, r * 1.5);
        shadowGrad.addColorStop(0, 'rgba(0, 0, 0, 0.65)');
        shadowGrad.addColorStop(1, 'rgba(0, 0, 0, 0)');
        ctx.fillStyle = shadowGrad;
        ctx.beginPath();
        ctx.ellipse(x, y + r, r * 1.1 * pulse, r * 0.35, 0, 0, 2 * Math.PI);
        ctx.fill();
        
        ctx.shadowBlur = 0;
        
        const img = this.images.ball;
        if (img.complete && img.naturalWidth !== 0) {
            ctx.drawImage(img, x - r * pulse, y - r * pulse, r * 2 * pulse, r * 2 * pulse);
        } else {
            // Procedural soccer ball
            ctx.fillStyle = '#ffffff';
            ctx.beginPath();
            ctx.arc(x, y, r * pulse, 0, 2 * Math.PI);
            ctx.fill();
            
            ctx.strokeStyle = '#000000';
            ctx.lineWidth = 2.5;
            ctx.stroke();
            
            ctx.fillStyle = '#1e1e1e';
            ctx.beginPath();
            ctx.arc(x, y, r * 0.3 * pulse, 0, 2 * Math.PI);
            ctx.fill();
            
            ctx.beginPath();
            ctx.moveTo(x, y - r * 0.3 * pulse);
            ctx.lineTo(x, y - r * pulse);
            ctx.moveTo(x - r * 0.28 * pulse, y + r * 0.1 * pulse);
            ctx.lineTo(x - r * 0.85 * pulse, y + r * 0.45 * pulse);
            ctx.moveTo(x + r * 0.28 * pulse, y + r * 0.1 * pulse);
            ctx.lineTo(x + r * 0.85 * pulse, y + r * 0.45 * pulse);
            ctx.stroke();
        }
        ctx.restore();
    }

    drawAimReticle(ctx, zone) {
        const gl = this.physics.strikerGoalLeft;
        const gr = this.physics.strikerGoalRight;
        const gt = this.physics.strikerGoalTop;
        const gb = this.physics.strikerGoalBottom;
        const gw = gr - gl;
        const gh = gb - gt;
        
        ctx.save();
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.18)';
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        
        ctx.beginPath();
        // Vertical grid lines
        ctx.moveTo(gl + gw / 3, gt);
        ctx.lineTo(gl + gw / 3, gb);
        ctx.moveTo(gl + (2 * gw) / 3, gt);
        ctx.lineTo(gl + (2 * gw) / 3, gb);
        // Horizontal grid lines
        ctx.moveTo(gl, gt + gh / 3);
        ctx.lineTo(gr, gt + gh / 3);
        ctx.moveTo(gl, gt + (2 * gh) / 3);
        ctx.lineTo(gr, gt + (2 * gh) / 3);
        ctx.stroke();
        
        // Highlight active zone
        if (zone && this.physics.zones3d[zone]) {
            let colIndex = 1;
            if (zone.endsWith("LEFT")) colIndex = 0;
            else if (zone.endsWith("RIGHT")) colIndex = 2;
            
            let rowIndex = 1;
            if (zone.startsWith("TOP")) rowIndex = 0;
            else if (zone.startsWith("BOTTOM")) rowIndex = 2;
            
            const zx = gl + (colIndex * gw) / 3;
            const zy = gt + (rowIndex * gh) / 3;
            const zw = gw / 3;
            const zh = gh / 3;
            
            const fillGrad = ctx.createLinearGradient(zx, zy, zx, zy + zh);
            fillGrad.addColorStop(0, 'rgba(0, 240, 255, 0.25)');
            fillGrad.addColorStop(1, 'rgba(0, 100, 255, 0.08)');
            ctx.fillStyle = fillGrad;
            ctx.fillRect(zx, zy, zw, zh);
            
            ctx.strokeStyle = '#00f0ff';
            ctx.lineWidth = 3;
            ctx.setLineDash([]);
            ctx.strokeRect(zx + 2, zy + 2, zw - 4, zh - 4);
            
            ctx.fillStyle = '#00f0ff';
            ctx.font = 'bold 13px Outfit';
            ctx.textAlign = 'center';
            ctx.shadowBlur = 6;
            ctx.shadowColor = '#00f0ff';
            ctx.fillText('AIM TARGET', zx + zw / 2, zy + zh / 2 + 5);
            ctx.shadowBlur = 0;
        }
        ctx.restore();
    }

    // DRAWING HELPERS (CANVAS)
    renderProceduralStadium(ctx, w, h) {
        // Sky Gradient (Sunset to night transition)
        const skyH = Math.floor(h * 0.45);
        const skyGrad = ctx.createLinearGradient(0, 0, 0, skyH);
        skyGrad.addColorStop(0, '#0a0a1a'); // Dark night sky
        skyGrad.addColorStop(0.6, '#141432');
        skyGrad.addColorStop(1, '#1e2846');
        ctx.fillStyle = skyGrad;
        ctx.fillRect(0, 0, w, skyH);
        
        // Distant Stadium Seating
        ctx.fillStyle = '#0f141e';
        ctx.beginPath();
        ctx.moveTo(0, skyH);
        ctx.lineTo(w * 0.2, skyH * 0.7);
        ctx.lineTo(w * 0.8, skyH * 0.7);
        ctx.lineTo(w, skyH);
        ctx.fill();
        
        // Seating tiers
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
        ctx.lineWidth = 2;
        for (let i = 0; i < 5; i++) {
            ctx.beginPath();
            ctx.moveTo(0, skyH - i * 15);
            ctx.lineTo(w, skyH - i * 15);
            ctx.stroke();
        }
        
        // Floodlights
        const drawLight = (x, y) => {
            ctx.fillStyle = '#fff';
            ctx.beginPath();
            ctx.arc(x, y, 6, 0, Math.PI * 2);
            ctx.fill();
            
            ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
            ctx.beginPath();
            ctx.moveTo(x, y);
            ctx.lineTo(x - 60, skyH);
            ctx.lineTo(x + 60, skyH);
            ctx.fill();
        };
        drawLight(w * 0.2, skyH * 0.6);
        drawLight(w * 0.8, skyH * 0.6);
        
        // Advertising Boards
        ctx.fillStyle = '#1e1e28';
        ctx.fillRect(0, skyH - 10, w, 10);
        ctx.fillStyle = 'rgba(255, 255, 255, 0.1)';
        for (let i = 0; i < w; i += 80) {
            ctx.fillRect(i + 10, skyH - 8, 60, 6);
        }

        // Pitch Base Gradient
        const pitchH = h - skyH;
        const pitchGrad = ctx.createLinearGradient(0, skyH, 0, h);
        pitchGrad.addColorStop(0, '#0a4614');
        pitchGrad.addColorStop(1, '#1e8228');
        ctx.fillStyle = pitchGrad;
        ctx.fillRect(0, skyH, w, pitchH);
        
        // Perspective Mowing Stripes
        const numStripes = 10;
        for (let i = 0; i < numStripes; i++) {
            if (i % 2 === 0) {
                ctx.fillStyle = 'rgba(0, 0, 0, 0.12)';
                ctx.beginPath();
                
                // Vanishing point perspective
                const topW = w / numStripes;
                const bottomW = (w * 1.5) / numStripes;
                const topX = (i * topW);
                const bottomX = (i * bottomW) - (w * 0.25);
                
                ctx.moveTo(topX, skyH);
                ctx.lineTo(topX + topW, skyH);
                ctx.lineTo(bottomX + bottomW, h);
                ctx.lineTo(bottomX, h);
                ctx.fill();
            }
        }
        
        // Pitch Markings (Lines)
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
        ctx.lineWidth = 4;
        
        // Penalty Box
        ctx.beginPath();
        ctx.moveTo(this.physics.goalLeft - 100, skyH);
        ctx.lineTo(this.physics.goalLeft - 250, h * 0.7);
        ctx.lineTo(this.physics.goalRight + 250, h * 0.7);
        ctx.lineTo(this.physics.goalRight + 100, skyH);
        ctx.stroke();
        
        // Six-yard Box
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(this.physics.goalLeft - 20, skyH);
        ctx.lineTo(this.physics.goalLeft - 60, h * 0.55);
        ctx.lineTo(this.physics.goalRight + 60, h * 0.55);
        ctx.lineTo(this.physics.goalRight + 20, skyH);
        ctx.stroke();
        
        // Penalty spot
        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        ctx.beginPath();
        // Adjust for perspective
        ctx.ellipse(w / 2, this.physics.penaltySpot.y, 8, 4, 0, 0, 2 * Math.PI);
        ctx.fill();
    }

    drawStadium() {
        const ctx = this.ctx;
        const img = this.images.stadium;
        
        if (img.complete && img.naturalWidth !== 0 && this.assetManager) {
            ctx.drawImage(img, 0, 0, this.canvas.width, this.canvas.height);
        } else {
            // Use cached procedural stadium
            ctx.drawImage(this.stadiumCanvas, 0, 0);
        }
        
        // Global ambient night lighting
        ctx.fillStyle = 'rgba(0, 10, 30, 0.15)';
        ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    }

    drawGoal() {
        const ctx = this.ctx;
        const img = this.images.goal;
        const isGK = (this.gameModeType === "GOALKEEPER");
        
        // Choose goal coordinates based on mode
        const gl = isGK ? this.physics.gkGoalLeft : this.physics.goalLeft;
        const gr = isGK ? this.physics.gkGoalRight : this.physics.goalRight;
        const gt = isGK ? this.physics.gkGoalTop : this.physics.goalTop;
        const gb = isGK ? this.physics.gkGoalBottom : this.physics.goalBottom;
        const gw = gr - gl;
        const gh = gb - gt;
        
        // Depth scale for 3D goal posts
        const depth = isGK ? 40 : 25;
        
        // Net ripple offset calculation
        const flashElapsed = this.screenFlashTimer ? Date.now() - this.screenFlashTimer : 9999;
        const rippleActive = flashElapsed < 700;
        
        if (img.complete && img.naturalWidth !== 0 && !isGK) {
            ctx.drawImage(img, gl, gt);
        } else {
            // Procedural 3D Goal Frame
            
            // Back net structure (darker)
            const netAlpha = isGK ? 0.35 : 0.25;
            ctx.strokeStyle = `rgba(220, 220, 220, ${netAlpha})`;
            ctx.lineWidth = isGK ? 2 : 1.5;
            
            const backL = gl + depth;
            const backR = gr - depth;
            const backT = gt + depth;
            
            // Draw Net Mesh
            const space = isGK ? 30 : 20;
            
            // Vertical mesh lines
            for (let x = backL; x <= backR; x += space) {
                ctx.beginPath();
                ctx.moveTo(x, backT);
                for (let y = backT; y <= gb; y += 10) {
                    let rippleOff = 0;
                    if (rippleActive) {
                        const distFromCenter = Math.abs(x - this.ballScreenPos.x);
                        const rippleAmt = Math.max(0, 35 - distFromCenter * 0.1) * Math.sin((y - backT) * 0.15 + flashElapsed * 0.03);
                        rippleOff = rippleAmt * (1.0 - flashElapsed / 700);
                    }
                    ctx.lineTo(x + rippleOff, y);
                }
                ctx.stroke();
            }
            
            // Horizontal mesh lines
            for (let y = backT; y <= gb; y += space) {
                ctx.beginPath();
                ctx.moveTo(backL, y);
                for (let x = backL; x <= backR; x += 10) {
                    let rippleOff = 0;
                    if (rippleActive) {
                        const distFromBall = Math.abs(y - this.ballScreenPos.y);
                        const rippleAmt = Math.max(0, 25 - distFromBall * 0.1) * Math.sin((x - backL) * 0.1 + flashElapsed * 0.03);
                        rippleOff = rippleAmt * (1.0 - flashElapsed / 700);
                    }
                    ctx.lineTo(x, y + rippleOff);
                }
                ctx.stroke();
            }
            
            // Side nets
            ctx.beginPath();
            ctx.moveTo(gl, gt);
            ctx.lineTo(backL, backT);
            ctx.lineTo(backL, gb);
            ctx.stroke();
            
            ctx.beginPath();
            ctx.moveTo(gr, gt);
            ctx.lineTo(backR, backT);
            ctx.lineTo(backR, gb);
            ctx.stroke();

            // Goal Posts (Front)
            const postGrad = ctx.createLinearGradient(gl, 0, gl + 15, 0);
            postGrad.addColorStop(0, '#ffffff');
            postGrad.addColorStop(0.5, '#e0e0e0');
            postGrad.addColorStop(1, '#aaaaaa');
            
            ctx.fillStyle = postGrad;
            
            // Left Post
            ctx.beginPath();
            ctx.rect(gl - 8, gt - 8, 16, gh + 8);
            ctx.fill();
            
            // Right Post
            const postGradR = ctx.createLinearGradient(gr - 15, 0, gr, 0);
            postGradR.addColorStop(0, '#aaaaaa');
            postGradR.addColorStop(0.5, '#e0e0e0');
            postGradR.addColorStop(1, '#ffffff');
            ctx.fillStyle = postGradR;
            ctx.beginPath();
            ctx.rect(gr - 8, gt - 8, 16, gh + 8);
            ctx.fill();
            
            // Crossbar
            const barGrad = ctx.createLinearGradient(0, gt - 15, 0, gt);
            barGrad.addColorStop(0, '#ffffff');
            barGrad.addColorStop(0.5, '#e0e0e0');
            barGrad.addColorStop(1, '#999999');
            ctx.fillStyle = barGrad;
            ctx.beginPath();
            ctx.rect(gl - 8, gt - 8, gw + 16, 16);
            ctx.fill();
            
            // Ground Shadow for Posts
            ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
            ctx.beginPath();
            ctx.ellipse(gl, gb, 20, 8, 0, 0, Math.PI * 2);
            ctx.ellipse(gr, gb, 20, 8, 0, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    drawGoalkeeper() {
        const ctx = this.ctx;
        
        let img = this.images.goalkeeperIdle;
        if (this.gkDiveState === 'LEFT_DIVE') {
            img = this.images.goalkeeperDiveLeft;
        } else if (this.gkDiveState === 'RIGHT_DIVE') {
            img = this.images.goalkeeperDiveRight;
        } else if (this.outcomeText && this.outcomeText.includes("SAVE")) {
            img = this.images.goalkeeperSave;
        }
        
        ctx.save();
        ctx.translate(this.keeperX, this.keeperY);
        if (this.keeperRotation !== 0) {
            ctx.rotate((this.keeperRotation * Math.PI) / 180.0);
        }
        
        if (img.complete && img.naturalWidth !== 0) {
            // Draw pre-scaled goalkeeper image centered
            ctx.drawImage(img, -65, -85, 130, 170);
        } else {
            // Procedural goalie avatar
            // Jersey
            ctx.fillStyle = '#e6c80a';
            ctx.beginPath();
            ctx.moveTo(-25, -20);
            ctx.lineTo(25, -20);
            ctx.lineTo(20, 40);
            ctx.lineTo(-20, 40);
            ctx.closePath();
            ctx.fill();
            
            // Jersey blue stripes
            ctx.fillStyle = '#1478dc';
            ctx.fillRect(-10, -20, 6, 60);
            ctx.fillRect(4, -20, 6, 60);
            
            // Head
            ctx.fillStyle = '#f0c8a0';
            ctx.beginPath();
            ctx.arc(0, -40, 18, 0, 2 * Math.PI);
            ctx.fill();
            
            // Hair
            ctx.fillStyle = '#1e1e1e';
            ctx.beginPath();
            ctx.arc(0, -45, 18, Math.PI, 2 * Math.PI);
            ctx.fill();
            
            // Arms
            ctx.lineWidth = 12;
            ctx.strokeStyle = '#e6c80a';
            ctx.lineCap = 'round';
            // left arm
            ctx.beginPath();
            ctx.moveTo(-25, -15);
            ctx.lineTo(-60, -30);
            ctx.stroke();
            ctx.strokeStyle = '#f0c8a0';
            ctx.lineWidth = 10;
            ctx.beginPath();
            ctx.moveTo(-55, -28);
            ctx.lineTo(-85, -40);
            ctx.stroke();
            ctx.fillStyle = '#dc1e1e'; // red glove
            ctx.beginPath();
            ctx.arc(-88, -42, 12, 0, 2 * Math.PI);
            ctx.fill();
            
            ctx.strokeStyle = '#e6c80a';
            ctx.lineWidth = 12;
            ctx.beginPath();
            ctx.moveTo(25, -15);
            ctx.lineTo(60, -30);
            ctx.stroke();
            ctx.strokeStyle = '#f0c8a0';
            ctx.lineWidth = 10;
            ctx.beginPath();
            ctx.moveTo(55, -28);
            ctx.lineTo(85, -40);
            ctx.stroke();
            ctx.fillStyle = '#dc1e1e';
            ctx.beginPath();
            ctx.arc(88, -42, 12, 0, 2 * Math.PI);
            ctx.fill();
            
            ctx.fillStyle = '#282828';
            ctx.fillRect(-22, 40, 44, 25);
            
            ctx.fillStyle = '#0a0a0a';
            ctx.fillRect(-23, 75, 18, 12);
            ctx.fillRect(5, 75, 18, 12);
        }
        ctx.restore();
    }

    drawBall() {
        const ctx = this.ctx;
        const img = this.images.ball;
        const isGK = (this.gameModeType === "GOALKEEPER");
        
        // 1. Draw motion trail (motion-blurred copies of ball sprite/image)
        if (this.ballHistory && this.ballHistory.length > 1 && (this.gameState === "BALL_FLIGHT")) {
            for (let i = 0; i < this.ballHistory.length; i++) {
                const h = this.ballHistory[i];
                // Fades out for older frames (lower indices)
                const alpha = (i / this.ballHistory.length) * 0.28;
                ctx.save();
                ctx.globalAlpha = alpha;
                ctx.translate(h.x, h.y);
                
                // Spin rotation for trail points (slightly less rotation for older frames)
                const trailSpin = (this.ballSpinAngle || 0) * (i / this.ballHistory.length);
                ctx.rotate(trailSpin);
                
                if (img.complete && img.naturalWidth !== 0) {
                    ctx.drawImage(img, -h.r, -h.r, h.r * 2, h.r * 2);
                } else {
                    // Fallback white circle
                    ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
                    ctx.beginPath();
                    ctx.arc(0, 0, h.r, 0, 2 * Math.PI);
                    ctx.fill();
                }
                ctx.restore();
            }
        }
        
        // 2. Ground shadow — projected below the ball at pitch level
        if (this.gameState === "BALL_FLIGHT") {
            const shadowY = isGK ? this.physics.gkGoalBottom : this.physics.penaltySpot.y;
            
            // Height of the ball off the ground (in pixels)
            const heightOffGround = Math.max(0, shadowY - this.ballScreenPos.y);
            
            // As height increases: shadow is more diffused, larger, and fainter
            const diffusionFactor = 1.0 + (heightOffGround / 150.0);
            const opacityFactor = Math.max(0.05, 0.4 - (heightOffGround / 400.0));
            
            // Scale shadow base alpha by depth too (fades as it gets further/closer depending on mode)
            const zFactor = isGK ? Math.max(0.2, Math.min(1.0, this.ballPos3d[2])) : Math.max(0.2, 1.0 - (this.ballPos3d[2] || 0));
            const finalOpacity = opacityFactor * zFactor;
            
            const shadowR = this.ballRadius * diffusionFactor;
            
            ctx.save();
            ctx.translate(this.ballScreenPos.x, shadowY);
            ctx.scale(1.4, 0.35); // Elliptical shadow shape
            
            // Create a radial gradient for soft edges
            const shadowGrad = ctx.createRadialGradient(0, 0, shadowR * 0.1, 0, 0, shadowR);
            shadowGrad.addColorStop(0, `rgba(0, 0, 0, ${finalOpacity})`);
            shadowGrad.addColorStop(1, 'rgba(0, 0, 0, 0)');
            
            ctx.fillStyle = shadowGrad;
            ctx.beginPath();
            ctx.arc(0, 0, shadowR, 0, 2 * Math.PI);
            ctx.fill();
            ctx.restore();
        }
        
        ctx.save();
        ctx.translate(this.ballScreenPos.x, this.ballScreenPos.y);
        
        // Apply spin rotation
        const spinAngle = this.ballSpinAngle || 0;
        ctx.rotate(spinAngle);
        
        if (img.complete && img.naturalWidth !== 0) {
            ctx.drawImage(img, -this.ballRadius, -this.ballRadius, this.ballRadius * 2, this.ballRadius * 2);
        } else {
            // Procedural Ball — sphere with glossy highlight
            // Drop shadow
            ctx.fillStyle = 'rgba(20,20,20,0.3)';
            ctx.beginPath();
            ctx.arc(3, 3, this.ballRadius, 0, 2 * Math.PI);
            ctx.fill();
            
            // White sphere base
            ctx.fillStyle = '#f5f5f5';
            ctx.strokeStyle = '#282828';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(0, 0, this.ballRadius, 0, 2 * Math.PI);
            ctx.fill();
            ctx.stroke();
            
            // Glossy highlight
            const highlight = ctx.createRadialGradient(-this.ballRadius * 0.3, -this.ballRadius * 0.35, 1, 0, 0, this.ballRadius);
            highlight.addColorStop(0, 'rgba(255,255,255,0.65)');
            highlight.addColorStop(0.5, 'rgba(255,255,255,0.0)');
            highlight.addColorStop(1, 'rgba(0,0,0,0.0)');
            ctx.fillStyle = highlight;
            ctx.beginPath();
            ctx.arc(0, 0, this.ballRadius, 0, 2 * Math.PI);
            ctx.fill();
            
            // Black pentagon patches
            ctx.fillStyle = '#232323';
            ctx.beginPath();
            ctx.arc(0, 0, this.ballRadius * 0.32, 0, 2 * Math.PI);
            ctx.fill();
            
            for (let i = 0; i < 5; i++) {
                const angle = (i * 72 * Math.PI) / 180.0;
                const bx = Math.cos(angle) * (this.ballRadius * 0.68);
                const by = Math.sin(angle) * (this.ballRadius * 0.68);
                ctx.beginPath();
                ctx.arc(bx, by, this.ballRadius * 0.18, 0, 2 * Math.PI);
                ctx.fill();
                
                ctx.strokeStyle = '#282828';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(0, 0);
                ctx.lineTo(bx, by);
                ctx.stroke();
            }
        }
        ctx.restore();
    }

    drawGlove(ctx, x, y, isLeft) {
        ctx.save();
        ctx.translate(x, y);
        
        const scale = 1.0;
        ctx.scale(isLeft ? -scale : scale, scale); // Mirror left glove
        
        // Glow effect
        ctx.shadowBlur = 15;
        ctx.shadowColor = isLeft ? 'rgba(255, 50, 150, 0.8)' : 'rgba(50, 200, 255, 0.8)';
        ctx.strokeStyle = isLeft ? '#ff3296' : '#32c8ff';
        ctx.lineWidth = 3.5;
        
        // Base plate (glove back)
        ctx.fillStyle = 'rgba(15, 15, 25, 0.95)';
        ctx.beginPath();
        // Draw cuff
        ctx.moveTo(-20, 35);
        ctx.lineTo(20, 35);
        // Outer palm
        ctx.lineTo(25, 10);
        // Pinky finger
        ctx.lineTo(22, -25);
        ctx.lineTo(12, -25);
        // Ring finger
        ctx.lineTo(10, -35);
        ctx.lineTo(2, -35);
        // Middle finger
        ctx.lineTo(0, -40);
        ctx.lineTo(-8, -40);
        // Index finger
        ctx.lineTo(-10, -32);
        ctx.lineTo(-18, -32);
        // Inner palm to thumb
        ctx.lineTo(-20, -10);
        ctx.lineTo(-35, -5);
        ctx.lineTo(-30, 10);
        ctx.lineTo(-22, 10);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
        
        // Neon Accent lines on back of hand
        ctx.shadowBlur = 5;
        ctx.strokeStyle = isLeft ? 'rgba(255, 50, 150, 0.8)' : 'rgba(50, 200, 255, 0.8)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        // Center pad
        ctx.moveTo(-10, 15);
        ctx.lineTo(10, 15);
        ctx.lineTo(5, -5);
        ctx.lineTo(-5, -5);
        ctx.closePath();
        ctx.stroke();
        
        // Knuckle guards
        ctx.fillStyle = isLeft ? '#ff3296' : '#32c8ff';
        ctx.fillRect(-12, -15, 4, 8);
        ctx.fillRect(-4, -18, 4, 8);
        ctx.fillRect(4, -18, 4, 8);
        ctx.fillRect(12, -15, 4, 8);
        
        // Cuff band
        ctx.fillStyle = isLeft ? '#ff3296' : '#32c8ff';
        ctx.fillRect(-15, 25, 30, 6);
        
        ctx.restore();
    }

    drawGloves() {
        this.drawGlove(this.ctx, this.leftGlovePos.x, this.leftGlovePos.y, true);
        this.drawGlove(this.ctx, this.rightGlovePos.x, this.rightGlovePos.y, false);
    }

    drawGoalieSkeletalOverlay() {
        const lh = this.vision.smoothedJoints['left_hip'];
        const rh = this.vision.smoothedJoints['right_hip'];
        if (!lh || !rh) return;
        
        const ctx = this.ctx;
        const shWidth = this.calibrationData.shoulderWidth || 0.15;
        const armSpan = shWidth * 2.2;
        
        const bodyCenterX = (lh.x_norm + rh.x_norm) / 2.0;
        const bodyY = (lh.y_norm + rh.y_norm) / 2.0;
        
        const goalW = this.physics.goalWidth;
        const goalH = this.physics.goalHeight;
        
        // Draw skeletal line overlay inside goal center to assist alignment
        ctx.lineWidth = 3;
        ctx.strokeStyle = 'rgba(0, 255, 200, 0.6)';
        ctx.shadowBlur = 15;
        ctx.shadowColor = 'rgba(0, 255, 200, 0.4)';
        
        for (const conn of this.vision.connections) {
            const startJoint = this.vision.smoothedJoints[conn.start];
            const endJoint = this.vision.smoothedJoints[conn.end];
            
            if (startJoint && endJoint) {
                // Map offset from body center to goal coordinates
                const dxS = startJoint.x_norm - bodyCenterX;
                const dyS = bodyY - startJoint.y_norm;
                const dxE = endJoint.x_norm - bodyCenterX;
                const dyE = bodyY - endJoint.y_norm;
                
                const sensX = 1.3 / armSpan;
                const sensY = 1.4 / armSpan;
                
                const sx = Math.floor(this.physics.goalCenterX + dxS * sensX * goalW);
                const sy = Math.floor(this.physics.goalBottom - (dyS + 0.1) * sensY * goalH);
                
                const ex = Math.floor(this.physics.goalCenterX + dxE * sensX * goalW);
                const ey = Math.floor(this.physics.goalBottom - (dyE + 0.1) * sensY * goalH);
                
                ctx.beginPath();
                ctx.moveTo(sx, sy);
                ctx.lineTo(ex, ey);
                ctx.stroke();
            }
        }
        ctx.shadowBlur = 0;
    }

    // ── TRAINING MODE OVERLAY ─────────────────────────────────────────────
    // Shows biomechanical vectors, predicted target, and keeper colliders
    drawTrainingOverlay(ctx) {
        const gl = this.physics.strikerGoalLeft;
        const gr = this.physics.strikerGoalRight;
        const gt = this.physics.strikerGoalTop;
        const gb = this.physics.strikerGoalBottom;
        const gw = gr - gl;
        const gh = gb - gt;
        
        ctx.save();
        
        // \u2500\u2500 1. Semi-transparent HUD background panel \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        ctx.fillStyle = 'rgba(0,0,0,0.55)';
        ctx.beginPath();
        if (typeof ctx.roundRect === 'function') ctx.roundRect(12, 100, 230, 200, 10);
        else ctx.rect(12, 100, 230, 200);
        ctx.fill();
        ctx.strokeStyle = 'rgba(100,255,200,0.5)';
        ctx.lineWidth = 1.5;
        ctx.stroke();
        
        // Header
        ctx.fillStyle = '#64ffc8';
        ctx.font = 'bold 13px Share Tech Mono';
        ctx.textAlign = 'left';
        ctx.fillText('⚽ TRAINING MODE  [T]', 22, 120);
        
        // Kick phase bar
        const phaseColors = { IDLE: '#888', BACKSWING: '#ffe650', STRIKE: '#ff6400', FOLLOW_THROUGH: '#64ff64' };
        const phaseColor = phaseColors[this.kickPhase] || '#888';
        ctx.fillStyle = '#aaa';
        ctx.font = '11px Share Tech Mono';
        ctx.fillText('PHASE:', 22, 140);
        ctx.fillStyle = phaseColor;
        ctx.font = 'bold 11px Share Tech Mono';
        ctx.fillText(this.kickPhase, 80, 140);
        
        // Dominant leg
        const legColor = this.dominantLeg === 'left' ? '#ff9f43' : '#54a0ff';
        ctx.fillStyle = '#aaa'; ctx.font = '11px Share Tech Mono';
        ctx.fillText('DOM LEG:', 22, 157);
        ctx.fillStyle = legColor; ctx.font = 'bold 11px Share Tech Mono';
        ctx.fillText((this.dominantLeg || '?').toUpperCase(), 88, 157);
        
        // Foot velocity + acceleration
        const vel = Math.max(this.lastLeftVel, this.lastRightVel);
        const accel = this._liveFootAccel || 0;
        ctx.fillStyle = '#aaa'; ctx.font = '11px Share Tech Mono';
        ctx.fillText(`VEL: ${vel.toFixed(3)}`, 22, 174);
        ctx.fillText(`ACCEL: ${accel.toFixed(3)}`, 22, 191);
        
        // Swing vector display
        const bio = this._biomechSnapshot;
        if (bio) {
            ctx.fillStyle = '#aaa';
            ctx.fillText(`SWING X: ${bio.shotHorizontal.toFixed(3)}`, 22, 208);
            ctx.fillText(`HEIGHT R: ${bio.swingHeightRatio.toFixed(2)}`, 22, 225);
            ctx.fillText(`UP COMP: ${bio.upwardComponent.toFixed(3)}`, 22, 242);
        }
        
        // \u2500\u2500 2. Draw predicted shot target zone (highlighted in goal) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        if (this.kickInfo && this.kickInfo.targetZone && this.physics.zones3d[this.kickInfo.targetZone]) {
            const zone = this.kickInfo.targetZone;
            let colIdx = 1, rowIdx = 1;
            if (zone.endsWith('LEFT')) colIdx = 0;
            else if (zone.endsWith('RIGHT')) colIdx = 2;
            if (zone.startsWith('TOP')) rowIdx = 0;
            else if (zone.startsWith('BOTTOM')) rowIdx = 2;
            
            const zx = gl + (colIdx * gw) / 3;
            const zy = gt + (rowIdx * gh) / 3;
            const zw = gw / 3;
            const zh = gh / 3;
            
            ctx.strokeStyle = 'rgba(255,100,50,0.9)';
            ctx.lineWidth = 3;
            ctx.setLineDash([5, 4]);
            ctx.strokeRect(zx + 3, zy + 3, zw - 6, zh - 6);
            ctx.setLineDash([]);
            ctx.fillStyle = 'rgba(255,100,50,0.12)';
            ctx.fillRect(zx + 3, zy + 3, zw - 6, zh - 6);
            ctx.fillStyle = 'rgba(255,100,50,0.9)';
            ctx.font = 'bold 12px Outfit';
            ctx.textAlign = 'center';
            ctx.shadowBlur = 6;
            ctx.shadowColor = '#ff6432';
            ctx.fillText('SHOT', zx + zw / 2, zy + zh / 2 + 4);
            ctx.shadowBlur = 0;
        }
        
        // \u2500\u2500 3. Draw goalkeeper physical colliders \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        if (this._keeperColliderDef && this._keeperColliderDef.length > 0) {
            const kx = this.keeperX;
            const ky = this.keeperY;
            const colColors = { head: '#ff6464', torso: '#ffa040', left_arm: '#40a0ff', right_arm: '#40a0ff' };
            
            this._keeperColliderDef.forEach(c => {
                const cx = kx + c.ox - c.w / 2;
                const cy = ky + c.oy - c.h / 2;
                ctx.strokeStyle = colColors[c.id] || '#ffffff';
                ctx.lineWidth = 2;
                ctx.setLineDash([4, 3]);
                ctx.strokeRect(cx, cy, c.w, c.h);
                ctx.setLineDash([]);
                ctx.fillStyle = (colColors[c.id] || '#ffffff').replace(')', ',0.08)').replace('rgb', 'rgba');
                ctx.fillRect(cx, cy, c.w, c.h);
                
                ctx.fillStyle = colColors[c.id] || '#ffffff';
                ctx.font = '9px Share Tech Mono';
                ctx.textAlign = 'center';
                ctx.fillText(c.id.toUpperCase(), cx + c.w / 2, cy + c.h / 2 + 3);
            });
        }
        
        // \u2500\u2500 4. Biomechanical leg vector arrows (live) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        const joints = this.vision.smoothedJoints;
        const activeLeg = this.dominantLeg || 'right';
        const hip   = joints[`${activeLeg}_hip`];
        const knee  = joints[`${activeLeg}_knee`];
        const ankle = joints[`${activeLeg}_ankle`];
        
        if (hip && ankle) {
            const cw = this.canvas.width, ch = this.canvas.height;
            const toScreen = (j) => ({ x: (1 - j.x_norm) * cw, y: j.y_norm * ch }); // mirrored
            const hipS   = toScreen(hip);
            const kneeS  = knee ? toScreen(knee) : { x: (hipS.x + toScreen(ankle).x)/2, y: (hipS.y + toScreen(ankle).y)/2 };
            const ankleS = toScreen(ankle);
            
            const drawArrow = (x1, y1, x2, y2, color, label) => {
                ctx.save();
                ctx.strokeStyle = color;
                ctx.lineWidth = 3;
                ctx.shadowBlur = 8;
                ctx.shadowColor = color;
                ctx.beginPath();
                ctx.moveTo(x1, y1);
                ctx.lineTo(x2, y2);
                ctx.stroke();
                
                // Arrowhead
                const ang = Math.atan2(y2 - y1, x2 - x1);
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.moveTo(x2, y2);
                ctx.lineTo(x2 - 10 * Math.cos(ang - 0.4), y2 - 10 * Math.sin(ang - 0.4));
                ctx.lineTo(x2 - 10 * Math.cos(ang + 0.4), y2 - 10 * Math.sin(ang + 0.4));
                ctx.closePath();
                ctx.fill();
                
                ctx.shadowBlur = 0;
                ctx.font = 'bold 10px Share Tech Mono';
                ctx.textAlign = 'center';
                ctx.fillText(label, (x1+x2)/2 + 12, (y1+y2)/2);
                ctx.restore();
            };
            
            drawArrow(hipS.x, hipS.y, kneeS.x, kneeS.y, '#ffe650', 'HIP→KN');
            drawArrow(kneeS.x, kneeS.y, ankleS.x, ankleS.y, '#64ff64', 'KN→ANK');
            
            // Joint dots
            [[hipS, '#ffe650'], [kneeS, '#ff9f43'], [ankleS, '#64ff64']].forEach(([pt, col]) => {
                ctx.fillStyle = col;
                ctx.shadowBlur = 10;
                ctx.shadowColor = col;
                ctx.beginPath();
                ctx.arc(pt.x, pt.y, 7, 0, 2 * Math.PI);
                ctx.fill();
                ctx.shadowBlur = 0;
            });
        }
        
        // ── 5. Goal Collision Bounds & Impact Point ──
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 6]);
        ctx.strokeRect(gl, gt, gw, gh);
        ctx.setLineDash([]);
        
        ctx.fillStyle = '#00ff00';
        ctx.font = '10px Share Tech Mono';
        ctx.textAlign = 'center';
        ctx.fillText('PHYSICS GOAL BOUNDS', gl + gw / 2, gt - 5);
        
        if (this.ballScreenPos && (this.gameState === "BALL_FLIGHT" || this.gameState === "RESULT_CELEBRATION")) {
            const bx = Math.round(this.ballScreenPos.x);
            const by = Math.round(this.ballScreenPos.y);
            
            ctx.strokeStyle = '#ff00ff';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(bx - 10, by - 10);
            ctx.lineTo(bx + 10, by + 10);
            ctx.moveTo(bx + 10, by - 10);
            ctx.lineTo(bx - 10, by + 10);
            ctx.stroke();
            
            ctx.fillStyle = '#ff00ff';
            ctx.fillText('IMPACT', bx, by - 15);
        }
        
        // \u2500\u2500 5. Mode label badge \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        const pulse = 0.7 + 0.3 * Math.sin(Date.now() * 0.005);
        ctx.globalAlpha = pulse;
        ctx.fillStyle = 'rgba(100,255,200,0.9)';
        ctx.font = 'bold 11px Share Tech Mono';
        ctx.textAlign = 'right';
        ctx.fillText('TRAINING  [T to exit]', this.canvas.width - 14, 118);
        ctx.globalAlpha = 1.0;
        
        ctx.restore();
    }

    drawCornersWebcam() {
        const ctx = this.gameWebcamCtx;
        const w = this.gameWebcamCanvas.width;
        const h = this.gameWebcamCanvas.height;
        ctx.clearRect(0, 0, w, h);
        
        // Render webcam video feed
        this.vision.drawSkeleton(ctx, w, h, '#ff3296', true);
    }



    updateBroadcastOverlay() {
        const status = this.vision.getTrackingStatus();
        const dotEl = document.getElementById('broadcast-status-dot');
        const textEl = document.getElementById('broadcast-status-text');
        
        if (dotEl && textEl) {
            textEl.innerText = `TRACKING: ${status}`;
            dotEl.className = 'status-dot';
            if (status === 'GOOD') {
                dotEl.classList.add('green-pulse');
            } else if (status === 'PARTIAL') {
                dotEl.classList.add('yellow-pulse');
            } else {
                dotEl.classList.add('red-pulse');
            }
        }
        
        const fpsEl = document.getElementById('broadcast-fps');
        if (fpsEl) {
            fpsEl.innerText = `${this.fps} FPS`;
        }
    }

    updateDebugPanel() {
        const panel = document.getElementById('debug-panel');
        if (!panel) return;
        
        if (!this.debugMode) {
            panel.classList.remove('active');
            return;
        }
        
        panel.classList.add('active');
        
        const joints = this.vision.smoothedJoints;
        const lw = joints['left_wrist'];
        const rw = joints['right_wrist'];
        const la = joints['left_ankle'];
        const ra = joints['right_ankle'];
        
        const lwStr = lw ? `(${lw.x_norm.toFixed(2)}, ${lw.y_norm.toFixed(2)}) [Conf: ${lw.visibility.toFixed(2)}]` : 'N/A';
        const rwStr = rw ? `(${rw.x_norm.toFixed(2)}, ${rw.y_norm.toFixed(2)}) [Conf: ${rw.visibility.toFixed(2)}]` : 'N/A';
        const laStr = la ? `(${la.x_norm.toFixed(2)}, ${la.y_norm.toFixed(2)}) [Conf: ${la.visibility.toFixed(2)}]` : 'N/A';
        const raStr = ra ? `(${ra.x_norm.toFixed(2)}, ${ra.y_norm.toFixed(2)}) [Conf: ${ra.visibility.toFixed(2)}]` : 'N/A';
        
        const bodyCenter = this.calibrationData.bodyCenter || 0.5;
        
        let currentAction = 'NONE';
        if (this.state === "PLAY_STRIKER") {
            currentAction = this.gameState;
        } else if (this.state === "PLAY_GOALKEEPER") {
            currentAction = this.gameState;
            if (this.gkDiveState && (Date.now() - this.gkDiveTimer < 600)) {
                currentAction += ` (DIVE: ${this.gkDiveState})`;
            }
        } else if (this.state === "CALIBRATION") {
            currentAction = "CALIBRATION";
        }
        
        let kickVelStr = '0.00';
        if (this.kickInfo) {
            kickVelStr = this.kickInfo.velocity.toFixed(3);
        } else {
            const leftVel = this.lastLeftVel || 0.0;
            const rightVel = this.lastRightVel || 0.0;
            kickVelStr = `L: ${leftVel.toFixed(3)} | R: ${rightVel.toFixed(3)}`;
        }
        
        let gkDebugHtml = '';
        if (this.gameModeType === "GOALKEEPER" && this._gkDebug) {
            const d = this._gkDebug;
            gkDebugHtml = `
                <hr style="border-color:rgba(255,255,255,0.15);margin:4px 0">
                <strong style="color:#00e5ff">── GK MAPPING ──</strong><br>
                <strong>Raw L Wrist:</strong> (${d.rawLX}, ${d.rawLY})<br>
                <strong>Raw R Wrist:</strong> (${d.rawRX}, ${d.rawRY})<br>
                <strong>Map L→Goal:</strong> (${d.mapLX}, ${d.mapLY})<br>
                <strong>Map R→Goal:</strong> (${d.mapRX}, ${d.mapRY})<br>
                <strong>Neutral:</strong> X=${d.neutralX} Y=${d.neutralY}<br>
                <strong>Reach:</strong> top=${d.reachTop} bot=${d.reachBott}<br>
                <strong>Sensitivity:</strong> X=${d.xSens} Y=${d.ySens}<br>
                <strong>Reach% L/R:</strong> ${d.reachPctL} / ${d.reachPctR}
            `;
        }
        
        let strikerDebugHtml = '';
        if (this.gameModeType === "STRIKER") {
            const leftVel = this.lastLeftVel || 0.0;
            const rightVel = this.lastRightVel || 0.0;
            const footAccel = this._liveFootAccel || 0.0;
            let legText = 'N/A';
            let angleText = '0.0°';
            let orientText = '0.0°';
            let predText = 'N/A';
            let confText = '0%';
            let powerText = '0%';
            let zoneText = 'N/A';
            let swingVxText = '0.000';
            let swingVyText = '0.000';
            if (this.kickInfo) {
                legText = this.kickInfo.leg.toUpperCase();
                angleText = `${this.kickInfo.kick_angle.toFixed(1)}°`;
                orientText = `${this.kickInfo.body_orientation.toFixed(1)}°`;
                predText = this.kickInfo.prediction;
                confText = `${Math.round(this.kickInfo.confidence * 100)}%`;
                powerText = `${this.kickInfo.power}%`;
                zoneText = this.kickInfo.targetZone || 'N/A';
                swingVxText = (this.kickInfo.swingVx || 0).toFixed(3);
                swingVyText = (this.kickInfo.swingVy || 0).toFixed(3);
            }
            const domLeg = (this.dominantLeg || '?').toUpperCase();
            const domVotes = `L:${this.dominantLegVotes.left} R:${this.dominantLegVotes.right}`;
            const bsAnchor = this.backswingAnchor ? 'SET' : 'NONE';
            const colliderCount = (this._keeperColliderDef || []).length;
            const gl = this.physics.strikerGoalLeft;
            const gr = this.physics.strikerGoalRight;
            const gt = this.physics.strikerGoalTop;
            const gb = this.physics.strikerGoalBottom;
            const ballX = this.ballScreenPos ? Math.round(this.ballScreenPos.x) : 0;
            const ballY = this.ballScreenPos ? Math.round(this.ballScreenPos.y) : 0;
            const insideGoal = (ballX >= gl && ballX <= gr && ballY >= gt && ballY <= gb);
            
            const triggerIdx = this.aiDiveTriggerIndex || 0;
            const delay = Math.round(triggerIdx * 16.67);
            
            strikerDebugHtml = `
                <hr style="border-color:rgba(255,255,255,0.15);margin:4px 0">
                <strong style="color:#64ff64">── BIOMECH STRIKER ──</strong><br>
                <strong>Dom Leg:</strong> ${domLeg} (${domVotes})<br>
                <strong>Left Vel:</strong> ${leftVel.toFixed(3)}<br>
                <strong>Right Vel:</strong> ${rightVel.toFixed(3)}<br>
                <strong>Foot Accel:</strong> ${footAccel.toFixed(3)}<br>
                <strong>Kick Phase:</strong> <span style="color:${this.kickPhase === 'IDLE' ? '#888' : '#ffe650'}">${this.kickPhase}</span><br>
                <strong>Backswing:</strong> ${bsAnchor}<br>
                <strong>Swing Vec:</strong> X=${swingVxText} Y=${swingVyText}<br>
                <strong>Curve Angle:</strong> ${angleText}<br>
                <strong>Body Orient:</strong> ${orientText}<br>
                <strong style="color:#ffae19">── GK PERCEPTION ──</strong><br>
                <strong>Actual Aim:</strong> ${this.currentAimZone}<br>
                <strong>GK Prediction:</strong> ${this.keeperPredictedTarget || 'N/A'}<br>
                <strong>GK Confidence:</strong> ${this.keeperPredictionConfidence}%<br>
                <strong>GK State:</strong> <span style="color:#00ffcc">${this.keeperAiState}</span><br>
                <strong>Reaction Delay:</strong> ${delay}ms<br>
                <hr style="border-color:rgba(255,255,255,0.15);margin:4px 0">
                <strong>Shot Target:</strong> ${zoneText}<br>
                <strong>Power:</strong> ${powerText}<br>
                <strong>AI Predict:</strong> ${predText} (${confText})<br>
                <strong>GK Colliders:</strong> ${colliderCount} parts<br>
                <strong>Ball Position:</strong> (${ballX}, ${ballY})<br>
                <strong>Inside Goal:</strong> ${insideGoal ? '<span style="color:#64ff64">TRUE</span>' : '<span style="color:#ff6464">FALSE</span>'}
            `;
        }
        
        const debugHtml = `
            <strong>FPS:</strong> ${this.fps}<br>
            <strong>Left Wrist:</strong> ${lwStr}<br>
            <strong>Right Wrist:</strong> ${rwStr}<br>
            <strong>Left Ankle:</strong> ${laStr}<br>
            <strong>Right Ankle:</strong> ${raStr}<br>
            <strong>Body Center:</strong> ${bodyCenter.toFixed(3)}<br>
            <strong>Kick Velocity:</strong> ${kickVelStr}<br>
            <strong>Current Action:</strong> ${currentAction}
            ${gkDebugHtml}
            ${strikerDebugHtml}
        `;
        
        document.getElementById('debug-content').innerHTML = debugHtml;
    }

    // UTILS
    runNameInputLoop() {}
    runLeaderboardLoop() {}
}

// Global initialization with readyState protection to prevent DOMContentLoaded race conditions
const initGame = () => {
    const game = new GameOrchestrator();
    game.start();
};

if (document.readyState === 'loading') {
    window.addEventListener('DOMContentLoaded', initGame);
} else {
    initGame();
}
