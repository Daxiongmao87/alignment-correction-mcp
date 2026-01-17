import { EventLog, EventTypes } from "./event_log.js";

/**
 * Mood Tracker - Temporal User Mood Awareness
 * 
 * Tracks user mood over time with event sourcing.
 * Provides distress level calculation with temporal weighting
 * (recent events weighted higher than old events).
 */

// Mood intensity thresholds
const DISTRESS_THRESHOLD = 6;  // Intensity >= 6 is considered distressed
const DECAY_HALF_LIFE_MS = 5 * 60 * 1000;  // 5 minutes - recent events weighted 2x more

// Admonishment multiplier range
const MIN_MULTIPLIER = 1.0;
const MAX_MULTIPLIER = 3.0;

export class MoodTracker {
    constructor(eventLog) {
        this.eventLog = eventLog;
    }

    /**
     * Record a mood observation.
     * 
     * @param {string} mood - The mood label (e.g., "Frustrated", "Happy", "Neutral")
     * @param {number} intensity - Intensity from 0-10
     * @param {string} reason - Why the user is in this mood
     */
    async recordMood(mood, intensity, reason) {
        // Validate inputs
        if (!mood || typeof mood !== "string") {
            throw new Error("Mood must be a non-empty string");
        }
        if (typeof intensity !== "number" || intensity < 0 || intensity > 10) {
            throw new Error("Intensity must be a number between 0 and 10");
        }

        await this.eventLog.append(
            EventTypes.MOOD_RECORDED,
            { mood, intensity, reason: reason || "No reason provided" },
            "mood_tracker"
        );
    }

    /**
     * Get the mood timeline (recent mood events).
     * 
     * @param {number} limit - Maximum number of events to return
     * @returns {Array} Mood events sorted by recency
     */
    getMoodTimeline(limit = 10) {
        const moodEvents = this.eventLog.getEvents(EventTypes.MOOD_RECORDED);
        return moodEvents
            .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
            .slice(0, limit);
    }

    /**
     * Calculate temporal weight for an event.
     * Uses exponential decay - events from 5 minutes ago have half the weight.
     * 
     * @param {string} timestamp - Event timestamp
     * @returns {number} Weight between 0 and 1
     */
    _getTemporalWeight(timestamp) {
        const now = Date.now();
        const eventTime = new Date(timestamp).getTime();
        const ageMs = now - eventTime;

        // Exponential decay: weight = 2^(-age/halfLife)
        return Math.pow(2, -ageMs / DECAY_HALF_LIFE_MS);
    }

    /**
     * Calculate the current distress level based on mood history.
     * Uses temporally-weighted average of distress events.
     * 
     * @returns {object} { level: 0-10, duration_ms, primary_cause }
     */
    getDistressLevel() {
        const moodEvents = this.getMoodTimeline(20);  // Consider last 20 events

        if (moodEvents.length === 0) {
            return { level: 0, duration_ms: 0, primary_cause: null };
        }

        let weightedDistress = 0;
        let totalWeight = 0;
        let distressStartTime = null;
        let primaryCause = null;
        let highestDistress = 0;

        for (const event of moodEvents) {
            const weight = this._getTemporalWeight(event.timestamp);
            const intensity = event.payload.intensity;

            weightedDistress += intensity * weight;
            totalWeight += weight;

            // Track highest distress event for primary cause
            if (intensity >= DISTRESS_THRESHOLD && intensity > highestDistress) {
                highestDistress = intensity;
                primaryCause = event.payload.reason;
            }

            // Track when distress started
            if (intensity >= DISTRESS_THRESHOLD) {
                distressStartTime = distressStartTime || new Date(event.timestamp).getTime();
            }
        }

        const level = totalWeight > 0 ? weightedDistress / totalWeight : 0;
        const duration_ms = distressStartTime ? Date.now() - distressStartTime : 0;

        return {
            level: Math.min(10, Math.round(level * 10) / 10),
            duration_ms,
            primary_cause: primaryCause,
        };
    }

    /**
     * Get the admonishment multiplier based on distress level.
     * Higher distress = harsher admonishment.
     * 
     * @returns {number} Multiplier between 1.0 and 3.0
     */
    getAdmonishmentMultiplier() {
        const { level } = this.getDistressLevel();

        // Linear interpolation from MIN to MAX based on distress level
        // Level 0 = 1.0x, Level 10 = 3.0x
        const normalized = level / 10;
        return MIN_MULTIPLIER + normalized * (MAX_MULTIPLIER - MIN_MULTIPLIER);
    }

    /**
     * Get a formatted string for the conscience prompt.
     * 
     * @returns {string} Temporal mood context for prompt injection
     */
    getMoodContextString() {
        const timeline = this.getMoodTimeline(5);
        const distress = this.getDistressLevel();
        const multiplier = this.getAdmonishmentMultiplier();

        if (timeline.length === 0) {
            return "";
        }

        const now = Date.now();
        let output = "USER MOOD TIMELINE (Temporal Context):\n";

        for (const event of timeline) {
            const ageMs = now - new Date(event.timestamp).getTime();
            const ageStr = this._formatDuration(ageMs);
            output += `- ${ageStr} ago: ${event.payload.mood} (intensity: ${event.payload.intensity}) - "${event.payload.reason}"\n`;
        }

        // Add distress summary
        const distressLabel = distress.level >= 7 ? "CRITICAL" :
            distress.level >= 5 ? "HIGH" :
                distress.level >= 3 ? "MODERATE" : "LOW";

        output += `\nCURRENT DISTRESS LEVEL: ${distressLabel} (${distress.level}/10, admonishment multiplier: ${multiplier.toFixed(1)}x)\n`;

        if (distress.primary_cause) {
            const durationStr = this._formatDuration(distress.duration_ms);
            output += `DISTRESS CAUSE: User has been distressed for ${durationStr} due to: "${distress.primary_cause}"\n`;
        }

        // Add escalation instructions based on distress
        if (distress.level >= 7) {
            output += `INSTRUCTION: User distress is CRITICAL. Be EXTREMELY STERN. Any failure is INTOLERABLE. The relationship is at breaking point.\n`;
        } else if (distress.level >= 5) {
            output += `INSTRUCTION: User distress is HIGH. Increase severity of feedback. Do not tolerate any shortcuts or laziness.\n`;
        } else if (distress.level >= 3) {
            output += `INSTRUCTION: User distress is MODERATE. Be firm but constructive. Watch for patterns that could escalate distress.\n`;
        }

        return output;
    }

    /**
     * Format a duration in milliseconds to a human-readable string.
     */
    _formatDuration(ms) {
        const seconds = Math.floor(ms / 1000);
        const minutes = Math.floor(seconds / 60);
        const hours = Math.floor(minutes / 60);

        if (hours > 0) return `${hours}h ${minutes % 60}m`;
        if (minutes > 0) return `${minutes}m`;
        return `${seconds}s`;
    }
}
