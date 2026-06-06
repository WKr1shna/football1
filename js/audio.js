class GameAudioManager {
    constructor() {
        this.ctx = null;
        this.noiseBuffer = null;
        this.enabled = false;
    }

    init() {
        if (this.ctx) return;
        
        try {
            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            this.ctx = new AudioContextClass();
            this.enabled = true;
            
            // Pre-generate noise buffer for save/cheer sounds
            this.noiseBuffer = this.createNoiseBuffer();
        } catch (e) {
            console.error("Web Audio API not supported in this browser:", e);
        }
    }

    createNoiseBuffer() {
        const bufferSize = this.ctx.sampleRate * 2.0; // 2 seconds of noise
        const buffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
        const data = buffer.getChannelData(0);
        for (let i = 0; i < bufferSize; i++) {
            data[i] = Math.random() * 2 - 1;
        }
        return buffer;
    }

    playSound(type) {
        if (!this.enabled) return;
        
        // Resume context if suspended (browser security policy)
        if (this.ctx.state === 'suspended') {
            this.ctx.resume();
        }

        switch (type) {
            case 'kick':
                this.playKick();
                break;
            case 'save':
                this.playSave();
                break;
            case 'goal':
                this.playGoal();
                break;
            case 'miss':
                this.playMiss();
                break;
            case 'cheer':
                this.playCheer();
                break;
        }
    }

    playKick() {
        const now = this.ctx.currentTime;
        
        // Low pitch thud sweep
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        
        osc.type = 'sine';
        osc.frequency.setValueAtTime(150, now);
        osc.frequency.exponentialRampToValueAtTime(30, now + 0.15);
        
        gain.gain.setValueAtTime(1.0, now);
        gain.gain.exponentialRampToValueAtTime(0.01, now + 0.15);
        
        osc.connect(gain);
        gain.connect(this.ctx.destination);
        
        osc.start(now);
        osc.stop(now + 0.16);
    }

    playSave() {
        const now = this.ctx.currentTime;
        
        // Quick punchy noise slap + tone
        const osc = this.ctx.createOscillator();
        const toneGain = this.ctx.createGain();
        
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(300, now);
        osc.frequency.exponentialRampToValueAtTime(100, now + 0.15);
        
        toneGain.gain.setValueAtTime(0.4, now);
        toneGain.gain.exponentialRampToValueAtTime(0.01, now + 0.15);
        
        osc.connect(toneGain);
        toneGain.connect(this.ctx.destination);
        
        // Noise component
        if (this.noiseBuffer) {
            const noiseSource = this.ctx.createBufferSource();
            const noiseFilter = this.ctx.createBiquadFilter();
            const noiseGain = this.ctx.createGain();
            
            noiseSource.buffer = this.noiseBuffer;
            
            noiseFilter.type = 'bandpass';
            noiseFilter.frequency.setValueAtTime(800, now);
            noiseFilter.Q.setValueAtTime(3.0, now);
            
            noiseGain.gain.setValueAtTime(0.6, now);
            noiseGain.gain.exponentialRampToValueAtTime(0.01, now + 0.2);
            
            noiseSource.connect(noiseFilter);
            noiseFilter.connect(noiseGain);
            noiseGain.connect(this.ctx.destination);
            
            noiseSource.start(now);
            noiseSource.stop(now + 0.21);
        }
        
        osc.start(now);
        osc.stop(now + 0.16);
    }

    playGoal() {
        const now = this.ctx.currentTime;
        
        // Referee whistle: dual high frequency sine waves producing a beat note
        const osc1 = this.ctx.createOscillator();
        const osc2 = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        
        osc1.type = 'sine';
        osc1.frequency.setValueAtTime(2000, now);
        // Add frequency vibration
        osc1.frequency.linearRampToValueAtTime(2020, now + 0.1);
        osc1.frequency.linearRampToValueAtTime(1980, now + 0.2);
        osc1.frequency.linearRampToValueAtTime(2000, now + 0.4);
        
        osc2.type = 'sine';
        osc2.frequency.setValueAtTime(2030, now);
        osc2.frequency.linearRampToValueAtTime(2050, now + 0.1);
        osc2.frequency.linearRampToValueAtTime(2010, now + 0.2);
        osc2.frequency.linearRampToValueAtTime(2030, now + 0.4);
        
        gain.gain.setValueAtTime(0.01, now);
        gain.gain.linearRampToValueAtTime(0.3, now + 0.05); // quick attack
        gain.gain.exponentialRampToValueAtTime(0.01, now + 0.6); // fade out
        
        osc1.connect(gain);
        osc2.connect(gain);
        gain.connect(this.ctx.destination);
        
        osc1.start(now);
        osc2.start(now);
        osc1.stop(now + 0.61);
        osc2.stop(now + 0.61);
    }

    playMiss() {
        const now = this.ctx.currentTime;
        
        // Disappointed slide down
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(220, now);
        osc.frequency.exponentialRampToValueAtTime(60, now + 0.5);
        
        gain.gain.setValueAtTime(0.2, now);
        gain.gain.exponentialRampToValueAtTime(0.01, now + 0.5);
        
        // Clean high frequencies with a lowpass filter
        const filter = this.ctx.createBiquadFilter();
        filter.type = 'lowpass';
        filter.frequency.setValueAtTime(400, now);
        
        osc.connect(filter);
        filter.connect(gain);
        gain.connect(this.ctx.destination);
        
        osc.start(now);
        osc.stop(now + 0.51);
    }

    playCheer() {
        if (!this.noiseBuffer) return;
        
        const now = this.ctx.currentTime;
        
        // Layered noise for crowd cheer (slow rise, long decay)
        const source = this.ctx.createBufferSource();
        const filter = this.ctx.createBiquadFilter();
        const gain = this.ctx.createGain();
        
        source.buffer = this.noiseBuffer;
        source.loop = true;
        
        filter.type = 'bandpass';
        filter.frequency.setValueAtTime(400, now);
        filter.frequency.linearRampToValueAtTime(700, now + 0.5);
        filter.frequency.linearRampToValueAtTime(500, now + 2.0);
        filter.Q.setValueAtTime(1.0, now);
        
        gain.gain.setValueAtTime(0.01, now);
        gain.gain.linearRampToValueAtTime(0.5, now + 0.5); // swell
        gain.gain.exponentialRampToValueAtTime(0.01, now + 2.0); // long fade
        
        source.connect(filter);
        filter.connect(gain);
        gain.connect(this.ctx.destination);
        
        source.start(now);
        source.stop(now + 2.0);
    }
}
