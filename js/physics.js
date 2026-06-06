class BallPhysics {
    constructor(screenWidth, screenHeight) {
        this.screenWidth = screenWidth;
        this.screenHeight = screenHeight;
        
        // ── STRIKER MODE — First-Person Penalty View ──────────────────────
        // Goal fills 72% of screen width, positioned near top (as seen from behind the ball)
        this.strikerGoalWidth  = Math.floor(screenWidth * 0.72);
        this.strikerGoalHeight = Math.floor(screenHeight * 0.52);
        this.strikerGoalLeft   = Math.floor((screenWidth - this.strikerGoalWidth) / 2);
        this.strikerGoalRight  = this.strikerGoalLeft + this.strikerGoalWidth;
        this.strikerGoalTop    = Math.floor(screenHeight * 0.10);
        this.strikerGoalBottom = this.strikerGoalTop + this.strikerGoalHeight;
        
        // Penalty spot (ball sits near bottom-center of screen, close to camera)
        this.penaltySpot = { x: screenWidth / 2, y: Math.floor(screenHeight * 0.86) };
        
        // Legacy aliases used by calibration-view, goalkeeper mode, mouse fallback etc.
        this.goalWidth  = this.strikerGoalWidth;
        this.goalHeight = this.strikerGoalHeight;
        this.goalLeft   = this.strikerGoalLeft;
        this.goalRight  = this.strikerGoalRight;
        this.goalTop    = this.strikerGoalTop;
        this.goalBottom = this.strikerGoalBottom;
        
        // ── GOALKEEPER MODE — Large First-Person View ────────────────────
        this.gkGoalWidth  = Math.floor(screenWidth * 0.76);
        this.gkGoalHeight = Math.floor(screenHeight * 0.58);
        this.gkGoalLeft   = Math.floor((screenWidth - this.gkGoalWidth) / 2);
        this.gkGoalRight  = this.gkGoalLeft + this.gkGoalWidth;
        this.gkGoalTop    = Math.floor(screenHeight * 0.15);
        this.gkGoalBottom = this.gkGoalTop + this.gkGoalHeight;
        
        this.goalCenterX = screenWidth / 2;
        
        // Ball sizing for striker: starts LARGE near camera, shrinks to small in goal
        this.startRadius = 42;  // large near player's feet (z=0)
        this.endRadius   = 9;   // small in the goal net (z=1)
        
        // 9 Goal Target Zones mapping (3D X & Y ranges)
        this.zones3d = {
            'TOP_LEFT':      { x: [-0.42, -0.16], y: [0.27, 0.37] },
            'TOP_CENTER':    { x: [-0.13, 0.13],  y: [0.27, 0.37] },
            'TOP_RIGHT':     { x: [0.16, 0.42],   y: [0.27, 0.37] },
            'MID_LEFT':      { x: [-0.42, -0.16], y: [0.14, 0.25] },
            'MID_CENTER':    { x: [-0.13, 0.13],  y: [0.14, 0.25] },
            'MID_RIGHT':     { x: [0.16, 0.42],   y: [0.14, 0.25] },
            'BOTTOM_LEFT':   { x: [-0.42, -0.16], y: [0.02, 0.12] },
            'BOTTOM_CENTER': { x: [-0.13, 0.13],  y: [0.02, 0.12] },
            'BOTTOM_RIGHT':  { x: [0.16, 0.42],   y: [0.02, 0.12] }
        };

        // 5 Goalkeeper columns mapping (visual boundaries in 3D X space)
        this.gkColumns3d = [
            { id: 'LEFT_DIVE',  range: [-0.60, -0.32] },
            { id: 'LEFT',       range: [-0.32, -0.10] },
            { id: 'CENTER',     range: [-0.10, 0.10] },
            { id: 'RIGHT',      range: [0.10, 0.32] },
            { id: 'RIGHT_DIVE', range: [0.32, 0.60] }
        ];
    }

    calculateTrajectory(zone, power, kickAngle = 0.0, isGoalkeeperMode = false) {
        const pStart = [0.0, 0.0, 0.0];
        
        // Retrieve target boundaries for selected zone (default to MID_CENTER)
        const zoneBounds = this.zones3d[zone] || this.zones3d['MID_CENTER'];
        
        // Generate random target coordinate inside the zone bounds
        const xBase = zoneBounds.x[0] + Math.random() * (zoneBounds.x[1] - zoneBounds.x[0]);
        const yBase = zoneBounds.y[0] + Math.random() * (zoneBounds.y[1] - zoneBounds.y[0]);
        
        // Add curve deflection to the final destination (shifting landing point)
        const curveOffset = Math.max(-0.25, Math.min(0.25, kickAngle / 90.0));
        const xTarget = xBase + curveOffset; // can go out-of-bounds (miss)
        const yTarget = yBase;
        
        const pEnd = [xTarget, yTarget, 1.0];
        
        // Add upward arc height based on zone height
        let arcHeight;
        if (zone.startsWith('TOP')) {
            arcHeight = 0.4 + Math.random() * 0.15;
        } else if (zone.startsWith('MID')) {
            arcHeight = 0.25 + Math.random() * 0.1;
        } else {
            arcHeight = 0.08 + Math.random() * 0.06;
        }
        
        // Bezier Control Point at Z = 0.5
        const pControl = [
            (pStart[0] + pEnd[0]) / 2.0 + curveOffset * 1.5,
            Math.max(pStart[1], pEnd[1]) + arcHeight,
            0.5
        ];
        
        // Calculate frames (power-based speed)
        const numFrames = Math.floor(Math.max(15, 60 - (power / 100.0) * 45));
        
        const trajectory = [];
        for (let i = 0; i <= numFrames; i++) {
            const t = i / numFrames;
            
            // Bezier formula
            const x3d = (1-t)*(1-t)*pStart[0] + 2*(1-t)*t*pControl[0] + t*t*pEnd[0];
            const y3d = (1-t)*(1-t)*pStart[1] + 2*(1-t)*t*pControl[1] + t*t*pEnd[1];
            const z3d = (1-t)*(1-t)*pStart[2] + 2*(1-t)*t*pControl[2] + t*t*pEnd[2];
            
            // Project 3D coordinate to 2D screen coordinate
            const screenPos = this.projectToScreen(x3d, y3d, z3d, isGoalkeeperMode);
            
            // Ball sizing
            let radius;
            if (isGoalkeeperMode) {
                radius = Math.floor(8 * (1.0 - z3d) + 55 * z3d);
            } else {
                // Striker: starts large near camera (z=0), shrinks as it flies into goal (z=1)
                radius = Math.floor(this.startRadius * (1.0 - z3d) + this.endRadius * z3d);
            }
            
            trajectory.push({
                x: screenPos.x,
                y: screenPos.y,
                z: z3d,
                x_3d: x3d,
                y_3d: y3d,
                radius: radius
            });
        }
        
        return trajectory;
    }

    projectToScreen(x_3d, y_3d, z_3d, isGoalkeeperMode = false) {
        if (isGoalkeeperMode) {
            // First-person Goalkeeper perspective
            const goalX = this.goalCenterX + (x_3d * this.gkGoalWidth);
            const goalY = this.gkGoalBottom - (y_3d * this.gkGoalHeight);
            
            const horizonX = this.goalCenterX;
            const horizonY = Math.floor(this.screenHeight * 0.38);
            
            const screenX = Math.floor(horizonX * (1.0 - z_3d) + goalX * z_3d);
            const screenY = Math.floor(horizonY * (1.0 - z_3d) + goalY * z_3d);
            
            return { x: screenX, y: screenY };
        } else {
            // ── Striker First-Person Perspective ──────────────────────────
            // z=0: ball is at penalty spot (bottom-center, near camera — large)
            // z=1: ball arrives at goal (top of screen — small)
            const goalX = this.goalCenterX + (x_3d * this.strikerGoalWidth * 0.5);
            const goalY = this.strikerGoalBottom - (y_3d * this.strikerGoalHeight);
            
            // Start position: ball is at penalty spot, slight lateral spread
            const startX = this.penaltySpot.x + (x_3d * this.screenWidth * 0.08);
            const startY = this.penaltySpot.y;
            
            const screenX = Math.floor(startX * (1.0 - z_3d) + goalX * z_3d);
            const screenY = Math.floor(startY * (1.0 - z_3d) + goalY * z_3d);
            
            return { x: screenX, y: screenY };
        }
    }



    // Determine which of the 5 goalkeeper columns a 3D coordinate X falls into
    getGKColumn(x_3d) {
        for (const col of this.gkColumns3d) {
            if (x_3d >= col.range[0] && x_3d <= col.range[1]) {
                return col.id;
            }
        }
        
        if (x_3d < -0.6) return 'LEFT_DIVE';
        if (x_3d > 0.6) return 'RIGHT_DIVE';
        return 'CENTER';
    }

    // Map hand offset ratio (relative to reach) to one of the 5 GK save columns
    getGKColumnFromOffset(offsetRatio) {
        if (offsetRatio < -0.65) return 'LEFT_DIVE';
        if (offsetRatio >= -0.65 && offsetRatio < -0.2) return 'LEFT';
        if (offsetRatio >= -0.2 && offsetRatio <= 0.2) return 'CENTER';
        if (offsetRatio > 0.2 && offsetRatio <= 0.65) return 'RIGHT';
        return 'RIGHT_DIVE';
    }
}
