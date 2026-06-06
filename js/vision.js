class GameVisionManager {
    constructor() {
        this.video = document.getElementById('webcam-video');
        this.pose = null;
        this.camera = null;
        
        this.webcamConnected = false;
        this.poseResults = null;
        this.onResultsCallback = null;
        
        this.smoothedJoints = {};
        
        // Define connections to draw skeleton lines manually with category type
        this.connections = [
            { start: 'left_shoulder', end: 'right_shoulder', type: 'torso' },
            { start: 'left_shoulder', end: 'left_hip', type: 'torso' },
            { start: 'right_shoulder', end: 'right_hip', type: 'torso' },
            { start: 'left_hip', end: 'right_hip', type: 'torso' },
            
            { start: 'left_shoulder', end: 'left_elbow', type: 'arms' },
            { start: 'left_elbow', end: 'left_wrist', type: 'arms' },
            { start: 'right_shoulder', end: 'right_elbow', type: 'arms' },
            { start: 'right_elbow', end: 'right_wrist', type: 'arms' },
            
            { start: 'left_hip', end: 'left_knee', type: 'legs' },
            { start: 'left_knee', end: 'left_ankle', type: 'legs' },
            { start: 'right_hip', end: 'right_knee', type: 'legs' },
            { start: 'right_knee', end: 'right_ankle', type: 'legs' }
        ];
    }

    async init(onResultsCallback) {
        this.onResultsCallback = onResultsCallback;
        
        try {
            // Check if MediaPipe Pose script loaded
            if (typeof window.Pose === 'undefined') {
                throw new Error("MediaPipe Pose library not loaded");
            }
            
            // 1. Initialize webcam stream with robust constraints
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                }
            });
            this.video.srcObject = stream;
            // Force playback start to avoid browser autoplay blocks
            await this.video.play().catch(err => console.warn("Video play promise catch:", err));
            
            // 2. Initialize Pose
            this.pose = new window.Pose({
                locateFile: (file) => {
                    return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
                }
            });
            
            this.pose.setOptions({
                modelComplexity: 1,
                smoothLandmarks: true,
                minDetectionConfidence: 0.6,
                minTrackingConfidence: 0.6
            });
            
            this.pose.onResults((results) => {
                this.poseResults = results;
                if (this.onResultsCallback) {
                    const joints = this.getKeyJoints(results);
                    this.onResultsCallback(joints, results);
                }
            });
            
            // Mark webcam as connected only after Pose tracker is successfully initialized
            this.webcamConnected = true;
            console.log("Webcam and MediaPipe successfully initialized.");
            
            // 3. Custom Frame Processing Loop using requestAnimationFrame
            let isProcessing = false;
            const processFrame = async () => {
                if (!this.webcamConnected) return;
                
                if (this.video.paused && !this.video.ended) {
                    this.video.play().catch(() => {});
                }
                
                // Only process frame if video has active data and is playing
                if (this.video.paused || this.video.ended || this.video.readyState < 2) {
                    requestAnimationFrame(processFrame);
                    return;
                }
                
                if (!isProcessing) {
                    isProcessing = true;
                    try {
                        await this.pose.send({ image: this.video });
                    } catch (err) {
                        console.error("MediaPipe Pose send error:", err);
                    }
                    isProcessing = false;
                }
                requestAnimationFrame(processFrame);
            };
            
            // Start the custom frame processor loop
            requestAnimationFrame(processFrame);
            
        } catch (e) {
            console.error("Camera or MediaPipe initialization failed. Switching to keyboard/mouse fallback mode.", e);
            this.webcamConnected = false;
        }
    }

    getKeyJoints(results) {
        const joints = {};
        if (!results || !results.poseLandmarks) return joints;
        
        const landmarks = results.poseLandmarks;
        
        // Anatomical mapping to MediaPipe Landmark indices
        const mapping = {
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28
        };
        
        for (const [name, index] of Object.entries(mapping)) {
            const lm = landmarks[index];
            if (lm) {
                // Confidence filtering: ignore landmarks with visibility < 0.6
                if (lm.visibility < 0.6) {
                    delete this.smoothedJoints[name];
                    continue;
                }

                // Mirror X coordinates for intuitive visual interaction
                const currentX = 1.0 - lm.x;
                const currentY = lm.y;
                const currentZ = lm.z;
                
                let smoothedX = currentX;
                let smoothedY = currentY;
                let smoothedZ = currentZ;
                
                // Temporal smoothing (exponential smoothing with alpha = 0.3)
                if (this.smoothedJoints[name]) {
                    const prev = this.smoothedJoints[name];
                    const alpha = 0.3;
                    smoothedX = alpha * currentX + (1 - alpha) * prev.x_norm;
                    smoothedY = alpha * currentY + (1 - alpha) * prev.y_norm;
                    smoothedZ = alpha * currentZ + (1 - alpha) * prev.z_norm;
                }
                
                const jointData = {
                    x_norm: smoothedX,
                    y_norm: smoothedY,
                    z_norm: smoothedZ,
                    visibility: lm.visibility
                };
                
                this.smoothedJoints[name] = jointData;
                joints[name] = jointData;
            } else {
                delete this.smoothedJoints[name];
            }
        }
        return joints;
    }

    getTrackingStatus() {
        if (!this.webcamConnected) return 'LOST';
        if (!this.poseResults || !this.poseResults.poseLandmarks) return 'LOST';
        
        const essential = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'];
        const details = ['left_wrist', 'right_wrist', 'left_ankle', 'right_ankle'];
        
        let essentialCount = 0;
        essential.forEach(name => {
            if (this.smoothedJoints[name]) essentialCount++;
        });
        
        let detailsCount = 0;
        details.forEach(name => {
            if (this.smoothedJoints[name]) detailsCount++;
        });
        
        if (essentialCount === 4 && detailsCount >= 3) {
            return 'GOOD';
        } else if (essentialCount >= 2 && detailsCount >= 1) {
            return 'PARTIAL';
        } else {
            return 'LOST';
        }
    }

    drawSkeleton(ctx, canvasWidth, canvasHeight, color = '#ff3296', drawVideo = false) {
        // Draw video background if specified (mirrored)
        if (drawVideo && this.webcamConnected && this.video.readyState >= 2) {
            ctx.save();
            ctx.translate(canvasWidth, 0);
            ctx.scale(-1, 1); // mirror video
            ctx.drawImage(this.video, 0, 0, canvasWidth, canvasHeight);
            ctx.restore();
        } else if (drawVideo) {
            // Fill black if no camera
            ctx.fillStyle = '#07070d';
            ctx.fillRect(0, 0, canvasWidth, canvasHeight);
            ctx.fillStyle = '#ff6464';
            ctx.font = '16px Outfit';
            ctx.textAlign = 'center';
            ctx.fillText("Webcam Disconnected - Mouse Fallback Enabled", canvasWidth / 2, canvasHeight / 2);
        }

        // Return if we don't have results yet
        if (!this.poseResults || !this.poseResults.poseLandmarks) return;
        
        const styles = {
            'torso': { color: '#32c8ff', glow: 'rgba(50, 200, 255, 0.5)' },
            'arms': { color: '#ff3296', glow: 'rgba(255, 50, 150, 0.5)' },
            'legs': { color: '#ffe650', glow: 'rgba(255, 230, 80, 0.5)' }
        };
        
        // 1. Draw connections
        ctx.lineWidth = 4;
        ctx.lineCap = 'round';
        
        for (const conn of this.connections) {
            const startJoint = this.smoothedJoints[conn.start];
            const endJoint = this.smoothedJoints[conn.end];
            
            if (startJoint && endJoint) {
                const style = styles[conn.type];
                ctx.strokeStyle = style.color;
                ctx.shadowBlur = 10;
                ctx.shadowColor = style.glow;
                
                const sx = startJoint.x_norm * canvasWidth;
                const sy = startJoint.y_norm * canvasHeight;
                const ex = endJoint.x_norm * canvasWidth;
                const ey = endJoint.y_norm * canvasHeight;
                
                ctx.beginPath();
                ctx.moveTo(sx, sy);
                ctx.lineTo(ex, ey);
                ctx.stroke();
            }
        }
        ctx.shadowBlur = 0;
        
        // 2. Draw key joints as neon circles
        ctx.fillStyle = '#ffffff';
        ctx.lineWidth = 2;
        
        for (const [name, joint] of Object.entries(this.smoothedJoints)) {
            let type = 'torso';
            if (name.includes('elbow') || name.includes('wrist')) type = 'arms';
            else if (name.includes('knee') || name.includes('ankle')) type = 'legs';
            
            const style = styles[type];
            ctx.strokeStyle = style.color;
            ctx.shadowBlur = 15;
            ctx.shadowColor = style.glow;
            
            const jx = joint.x_norm * canvasWidth;
            const jy = joint.y_norm * canvasHeight;
            
            ctx.beginPath();
            ctx.arc(jx, jy, 6, 0, 2 * Math.PI);
            ctx.fill();
            ctx.stroke();
        }
        
        // Reset canvas shadow states
        ctx.shadowBlur = 0;
    }
}
