import fs from "node:fs/promises";
import path from "path";
import crypto from "crypto";

/**
 * Event Log - SSR Event Sourcing Implementation
 * 
 * The Event Log is the Source of Truth. The runtime state (Constraint Store)
 * is merely a Materialized View (projection) derived from this log.
 * 
 * Per SSR §4.4: "SSR relies on Event Sourcing for replayability and consistency."
 */

const DEFAULT_EVENT_LOG_PATH = path.join(process.cwd(), "memory_event_log.json");

// Event Types per SSR §4.4.1
export const EventTypes = {
    CONSTRAINT_ADDED: "CONSTRAINT_ADDED",
    CONSTRAINT_UPDATED: "CONSTRAINT_UPDATED",
    CONSTRAINT_OBSOLETED: "CONSTRAINT_OBSOLETED",
    CONSTRAINT_CONTRADICTED: "CONSTRAINT_CONTRADICTED",
    // Mood tracking events
    MOOD_RECORDED: "MOOD_RECORDED",
};

/**
 * Generates a unique event ID.
 */
function generateEventId() {
    return `evt_${Date.now()}_${crypto.randomBytes(4).toString("hex")}`;
}

export class EventLog {
    constructor(filePath = DEFAULT_EVENT_LOG_PATH) {
        this.filePath = filePath;
        this.events = [];
    }

    /**
     * Load events from persistent storage.
     */
    async load() {
        try {
            const data = await fs.readFile(this.filePath, "utf-8");
            this.events = JSON.parse(data);
        } catch (error) {
            if (error.code !== "ENOENT") {
                console.error("Error loading event log:", error);
            }
            this.events = [];
        }
    }

    /**
     * Persist events to storage.
     */
    async save() {
        try {
            await fs.writeFile(this.filePath, JSON.stringify(this.events, null, 2));
        } catch (error) {
            console.error("Error saving event log:", error);
        }
    }

    /**
     * Append a new event to the log.
     * Per SSR: Events are immutable once written.
     * 
     * @param {string} eventType - One of EventTypes
     * @param {object} payload - Event-specific data
     * @param {string} source - Origin of the event (e.g., "conscience", "migration")
     * @returns {object} The created event
     */
    async append(eventType, payload, source = "conscience") {
        const event = {
            event_id: generateEventId(),
            timestamp: new Date().toISOString(),
            event_type: eventType,
            payload,
            source,
        };

        this.events.push(event);
        await this.save();
        return event;
    }

    /**
     * Get all events, optionally filtered by type.
     * 
     * @param {string} [eventType] - Filter by event type
     * @returns {Array} Events matching the filter
     */
    getEvents(eventType = null) {
        if (eventType) {
            return this.events.filter(e => e.event_type === eventType);
        }
        return [...this.events];
    }

    /**
     * Replay events to rebuild state.
     * Per SSR §4.4: "The Event Log is the Source of Truth; the Game World is merely its current projection."
     * 
     * @param {function} reducer - Function (state, event) => newState
     * @param {any} initialState - Starting state
     * @returns {any} Final state after replaying all events
     */
    replay(reducer, initialState) {
        return this.events.reduce(reducer, initialState);
    }

    /**
     * Get events since a specific event ID.
     * 
     * @param {string} sinceEventId - The event ID to start from (exclusive)
     * @returns {Array} Events after the specified ID
     */
    getEventsSince(sinceEventId) {
        const idx = this.events.findIndex(e => e.event_id === sinceEventId);
        if (idx === -1) {
            return [...this.events];
        }
        return this.events.slice(idx + 1);
    }
}
