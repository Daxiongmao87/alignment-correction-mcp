import { EventLog, EventTypes } from "./event_log.js";

/**
 * Constraint Store - SSR Constraint Graph Implementation
 * 
 * Per SSR §4.2: "Instead of a 'random seed,' unvisited worlds are defined by a Constraint Graph."
 * 
 * This store maintains user preferences and behavioral rules as constraints,
 * projecting canonical state from the Event Log.
 */

// Pruning thresholds per SSR §4.2.1
const STRENGTH_THRESHOLD = 0.15;

/**
 * Constraint Record format per SSR §4.2.1:
 * { key, value, strength, type: "hard"|"soft", source_event_id, ttl }
 */

export class ConstraintStore {
    constructor(eventLog) {
        this.eventLog = eventLog;
        this.constraints = new Map(); // key -> constraint record
    }

    /**
     * Rebuild the canonical state by replaying the Event Log.
     * Per SSR §3.1: "Canonical State: A Materialized View (Projection) derived from the Event Log."
     */
    rebuild() {
        this.constraints = this.eventLog.replay((state, event) => {
            switch (event.event_type) {
                case EventTypes.CONSTRAINT_ADDED: {
                    const { key, value, strength, type, ttl } = event.payload;
                    state.set(key, {
                        key,
                        value,
                        strength: strength ?? 1.0,
                        type: type ?? "hard",
                        source_event_id: event.event_id,
                        ttl: ttl ?? null,
                        created_at: event.timestamp,
                    });
                    break;
                }
                case EventTypes.CONSTRAINT_UPDATED: {
                    const { key, value, strength, type, ttl } = event.payload;
                    const existing = state.get(key);
                    if (existing) {
                        state.set(key, {
                            ...existing,
                            value: value ?? existing.value,
                            strength: strength ?? existing.strength,
                            type: type ?? existing.type,
                            ttl: ttl !== undefined ? ttl : existing.ttl,
                            source_event_id: event.event_id,
                        });
                    }
                    break;
                }
                case EventTypes.CONSTRAINT_OBSOLETED:
                case EventTypes.CONSTRAINT_CONTRADICTED: {
                    const { key } = event.payload;
                    state.delete(key);
                    break;
                }
            }
            return state;
        }, new Map());

        // Apply pruning policy
        this.prune();
    }

    /**
     * Prune expired or weak constraints.
     * Per SSR §4.2.1:
     * - Soft constraints dropped when strength < STRENGTH_THRESHOLD or ttl expires
     * - Hard constraints persist unless explicitly terminated
     */
    prune() {
        const now = Date.now();
        for (const [key, constraint] of this.constraints) {
            // Check TTL expiration
            if (constraint.ttl !== null) {
                const createdTime = new Date(constraint.created_at).getTime();
                const expiresAt = createdTime + (constraint.ttl * 1000); // ttl in seconds
                if (now > expiresAt) {
                    this.constraints.delete(key);
                    continue;
                }
            }

            // Check strength threshold for soft constraints only
            if (constraint.type === "soft" && constraint.strength < STRENGTH_THRESHOLD) {
                this.constraints.delete(key);
            }
        }
    }

    /**
     * Add a new constraint.
     * Validates and commits via Event Log.
     * 
     * @param {string} key - Unique identifier for the constraint
     * @param {string} value - The constraint content/rule
     * @param {object} options - { strength, type, ttl }
     */
    async add(key, value, options = {}) {
        const { strength = 1.0, type = "hard", ttl = null } = options;

        // Validation per SSR §4.3 - Engine validates before commit
        if (!key || typeof key !== "string") {
            throw new Error("Constraint key must be a non-empty string");
        }
        if (!value || typeof value !== "string") {
            throw new Error("Constraint value must be a non-empty string");
        }
        if (type !== "hard" && type !== "soft") {
            throw new Error("Constraint type must be 'hard' or 'soft'");
        }
        if (strength < 0 || strength > 1) {
            throw new Error("Constraint strength must be between 0 and 1");
        }

        const event = await this.eventLog.append(
            EventTypes.CONSTRAINT_ADDED,
            { key, value, strength, type, ttl }
        );

        // Update local state
        this.constraints.set(key, {
            key,
            value,
            strength,
            type,
            source_event_id: event.event_id,
            ttl,
            created_at: event.timestamp,
        });
    }

    /**
     * Update an existing constraint.
     */
    async update(key, updates) {
        if (!this.constraints.has(key)) {
            throw new Error(`Constraint '${key}' does not exist`);
        }

        await this.eventLog.append(
            EventTypes.CONSTRAINT_UPDATED,
            { key, ...updates }
        );

        this.rebuild();
    }

    /**
     * Mark a constraint as obsolete.
     * Per SSR §4.2.1: Hard constraints persist unless explicitly terminated.
     */
    async obsolete(key, reason = "explicitly removed") {
        if (!this.constraints.has(key)) {
            return; // Already gone
        }

        await this.eventLog.append(
            EventTypes.CONSTRAINT_OBSOLETED,
            { key, reason }
        );

        this.constraints.delete(key);
    }

    /**
     * Clear all constraints.
     */
    async clear() {
        const keys = Array.from(this.constraints.keys());
        for (const key of keys) {
            await this.obsolete(key, "bulk clear");
        }
    }

    /**
     * Get all active constraints.
     * @returns {Array} Array of constraint records
     */
    getAll() {
        return Array.from(this.constraints.values());
    }

    /**
     * Get constraints by type.
     * @param {string} type - "hard" or "soft"
     */
    getByType(type) {
        return this.getAll().filter(c => c.type === type);
    }

    /**
     * Get the canonical state as a formatted string for LLM context.
     * Replaces the old flat behavioral memory section.
     */
    getCanonicalStateString() {
        const constraints = this.getAll();
        if (constraints.length === 0) {
            return "";
        }

        return constraints.map(c => {
            const typeMarker = c.type === "hard" ? "[HARD]" : "[SOFT]";
            const strengthMarker = c.strength < 1.0 ? ` (strength: ${c.strength})` : "";
            return `${typeMarker}${strengthMarker} ${c.value}`;
        }).join("\n");
    }

    /**
     * Check if a constraint exists.
     */
    has(key) {
        return this.constraints.has(key);
    }

    /**
     * Get a specific constraint.
     */
    get(key) {
        return this.constraints.get(key);
    }
}
