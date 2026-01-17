# Semantic State Resolution (SSR)

## Architectural Standard for Just-In-Time Procedural Reality

**Author:** Patrick Richardson  
**Date:** December 19, 2025  
**Status:** Architectural Standard v1.0  
**License:** MIT

## 1. Executive Summary

Traditional game engines treat Narrative (Flavor Text, Visuals) and Systems (Stats, Mechanics) as separate domains. This separation leads to Ludonarrative Dissonance, where a sword described as "cursed" behaves identically to a standard iron blade.

Semantic State Resolution (SSR) is a unified architecture that resolves this issue by treating narrative and mechanics as two views of the same canonical state, revealed progressively through observation.

Unlike pure generative AI, SSR does not grant the Large Language Model (LLM) authority over the game state. Instead, the LLM acts as a Semantic Proposer within a strict, validated constraint-satisfaction loop. This ensures that while the content is infinite, the logic remains rigid, deterministic, and persistent.

## 2. Problem Statement: The Three Traps

Current procedural architectures suffer from three specific disconnects that prevent true immersion. SSR is explicitly designed to solve these architectural failures.

### 2.1 The Semantic Gap (The "Math vs. Meaning" Trap)

Standard procedural algorithms, such as Perlin Noise or RNG loot tables, are semantically blind.

**The Failure:** A generic "Level 5 Loot Table" drops items that mathematically fit the player's level but thematically contradict the context.

**Example:** A Fire Elemental drops a "Frost Wand" simply because the RNG rolled a valid ID, breaking immersion.

### 2.2 The Hallucination Trap (The "Dream Logic" Flaw)

Pure Generative AI lacks state rigidity. Without a canonical anchor, LLMs suffer from object impermanence.

**The Failure:** An NPC described as "mortally wounded" in Turn 1 might appear "perfectly healthy" in Turn 2 because the model context window shifted or the model hallucinated a repair.

**Result:** The world feels like a dream: fluid and reacting to the immediate prompt, but lacking causal history.

### 2.3 The Latency Trap (Static Fidelity)

To ensure coherence, traditional games must pre-simulate the entire world (e.g., Dwarf Fortress) or fully instantiate objects upon creation.

**The Failure:** The game must store the exact stats of a sword the moment it drops, even if the player never picks it up. This wastes resources on unobserved reality and prevents the world from reacting dynamically to player actions.

**Result:** High CPU/Memory overhead for low-interaction fidelity.

## 3. Core Definitions & Invariants

### 3.1 Core Terms

**Latent Entity (Superposition):**  
An entity that exists only as an opaque handle (ID) and a set of Constraints. It has no fully realized components (e.g., no stats, no mesh), only potential.

**Observation:**  
The event where a Player or System queries a property of a Latent Entity, triggering the resolution loop.

**Observation Scope:**

![Observation Scope](assets/ssr_observation_scope.png)

Defined strictly by the Interaction Horizon:
- **Visual Horizon:** Camera Frustum + Buffer
- **Logical Horizon:** Graph Distance (e.g., Depth 2 neighbors)

**Canonical State:**  
A Materialized View (Projection) derived from the Event Log. It serves as the fast runtime database but is not the ultimate source of truth.

### 3.2 Entity State Machine

The following diagram illustrates the immutable lifecycle of an SSR entity.

![Entity State Machine](assets/ssr_entity_state_machine.png)

### 3.3 The "No-Retcon" Invariant

**Invariant:** Once an entity resolves, its realized components become canonical and may only change through recorded Deltas caused by simulation or player action. The semantic model may propose, but the engine validates and commits. Unobserved entities remain latent, defined by constraints and provenance.

## 4. Architecture & Data Contracts

### 4.1 System Architecture

![System Architecture](assets/ssr_system_architecture.png)

The Engine acts as a Mediator to enforce the Strategy and Event Sourcing patterns.

### 4.2 The Constraint Store (The Constraint Graph)

Instead of a "random seed," unvisited worlds are defined by a Constraint Graph.

#### 4.2.1 Constraint Record

Each constraint is stored as a record:

```json
{ "key": "...", "value": "...", "strength": 0.5, "type": "hard|soft", "source_event_id": "...", "ttl": 10 }
```

**Pruning Policy:**
- **Soft constraints** are dropped when strength < Strength_Threshold (default 0.15) or when ttl expires.
- **Hard constraints** persist unless explicitly terminated by ConstraintObsoleted or ConstraintContradicted.

### 4.3 The Solver Interface (The Contract)

The boundary between the Game Engine and the LLM is strict.

1. **Engine Request:** Sends Context + Constraints + Schema + Validation Rules.
2. **LLM Proposal:** Returns structured JSON (Proposals).
3. **Engine Validation:** Checks schema, structural constraints, and game rules.
4. **Commit:** Validated data is written to the Event Log.

#### 4.3.1 Deterministic Fallback Protocol

If the LLM fails to produce a valid proposal within N retries, the system executes a Deterministic Fallback:

**Selection Mechanism:** `Index = Hash(Entity_ID + Hard_Constraints + Ruleset_Version) % Safety_Table_Size`

**Invariant:** The game loop must never block or crash due to Solver failure.

### 4.4 Persistence Model (Event Sourcing)

SSR relies on Event Sourcing for replayability and consistency. The Event Log is the Source of Truth; the Game World is merely its current projection.

#### 4.4.1 Event Schema

**ResolutionCommitted:** Writes the full component set.
- Payload: `{ components: { ... }, frozen_stats: { ... }, tags: [ ... ] }`

**DeltaApplied:** Writes patch operations (add/remove/set).
- Payload: `{ op: "set", path: "/components/Inventory/slots/main/durability", value: 12 }`

### 4.5 Architectural Pattern Mapping

SSR is a composite architecture built on standard software engineering patterns. This mapping ensures the system is maintainable and testable.

## 5. The SSR Lifecycle (The Loop)

The following sequence details the flow of data from Observation to Propagation, highlighting the strict separation between the Proposer (LLM) and the Truth (Event Log).

![SSR Lifecycle](assets/ssr_lifecycle.png)

**Phase 1: Superposition (Latent State)**
- The world contains Entity_ID_104.
- State: Latent
- Constraints: `{ Neighbor: "Industrial_Zone", Global_Event: "Acid_Rain" }`

**Phase 2: Observation (Trigger)**
- The player enters the sector containing Entity_ID_104.

**Phase 3: Proposal (Semantic Solving)**
- Engine query: "Propose a sector compatible with Industrial/Acid Rain context."
- LLM output: `{ Type: "Chemical_Runoff_Plant", Hazards: ["Corrosive_Puddles"] }`

**Phase 4: Validation & Resolution**
- Engine validates output against Structural Schemas + ruleset.
- Commit: ResolutionCommitted event written to log
- Materialize: Entity_ID_104 projected as Chemical Runoff Plant

**Phase 5: Propagation (Constraint Injection)**
- Engine injects constraints into neighbors via events.
- Action: Injects ConstraintInjected event for Entity_ID_105.
- Constraint: `Ground_Water: Contaminated`

**Note on Latency & Future Proofing**

The SSR architecture is Latency Agnostic. It utilizes Lookahead Buffers (resolving Depth+1 while the player is at Depth 0) to mask current LLM inference times. However, the architecture is designed to scale directly with model advances. As inference costs approach zero, the buffer size decreases, but the architectural invariants (Validation, Event Sourcing) remain identical.

## 6. Conflict Resolution & Precedence

![Conflict Resolution](assets/ssr_conflict_resolution.png)

When generating new content, constraints may conflict. The Solver follows this precedence hierarchy:

1. **Canonical State (Highest):** If a neighbor is already resolved, its reality is absolute.
2. **Hard Constraints:** Quest requirements or previous Propagation tags.
3. **Soft Constraints:** General themes.
4. **LLM Generation (Lowest):** The model's "creative filler."

**Conflict Policy:** If a Hard Constraint violates Canonical State, the system marks the Constraint as "Obsolete" or "False Rumor" rather than altering the map (Retconning).

## 7. Application Domains (Unified)

### 7.1 Progressive Mechanical Resolution (Loot)

1. **Drop:** Entity is Latent. Constraint: `Origin: Swamp_Beast`.
2. **Inspect:** Resolve Visuals. Result: "Slime-coated Blade." (Tags: Toxic, Crude).
3. **Equip:** Resolve Stats.

#### 7.1.1 The Translation Layer (Tag-to-Math)

The LLM is a Selector, not a Calculator. It generates Semantic Tags which map to Game Math via a Translation Table.

- **Schema Validation:** SSR validates **structure**, not **content**. The LLM may generate any tag that conforms to the schema's structural rules (format, regex patterns, type constraints). This preserves infinite creative variety while ensuring all outputs are mechanically valid.
- **No-Retcon Fix (Balance Patches):** Numeric baseline stats are Frozen at the moment of Resolution.

> [!IMPORTANT]
> **Schema vs. Whitelist:** SSR does NOT require enumerating every valid tag upfront. Instead, it validates that outputs conform to structural schemas (e.g., `^[A-Z][a-z]+(-[A-Z][a-z]+)*$` for tag format). The Translation Table maps **resolved tags** to math at runtime, not from a pre-defined finite list.

**The Maintenance Trade-off:** This approach deliberately trades Runtime Chaos for Design-Time Structure. The Translation Table defines math for common tags, with fallback rules for novel tags (e.g., semantic similarity matching, default stat blocks). This enables infinite generation while maintaining balanced gameplay.

### 7.2 Reverse Propagation (Cognition)

1. **Dialogue:** NPC generates a rumor: "The sewers are flooded."
2. **Propagation:** The Engine injects `State: Flooded` as a Hard Constraint into the unvisited Sewer Sector.
3. **Result:** When the player eventually reaches the sewers, the map generator must generate water obstacles to satisfy the constraint.

## 8. Conclusion

SSR is not merely "LLMs writing text." It is a rigorous system architecture where:

- **Latent Entities** minimize resource usage.
- **Event Sourcing** ensures deterministic replay and traceability.
- **Canonical Projections** prevent "dream logic."
- **Validation Layers** guarantee engine stability.

This standard provides the blueprint for infinite, coherent worlds that respect player agency and logical cause-and-effect.

## 9. Appendix: Reference Data Contracts

To facilitate implementation, the following JSON schemas define the core communication protocol between the Engine and the Semantic Solver.

### 9.1 Solver Request (Engine -> LLM)

```json
{
  "request_id": "req_8821a",
  "task_type": "RESOLVE_ZONE",
  "context": {
    "entity_id": "grid_4_5",
    "global_state": { "weather": "acid_rain", "alert": "high" },
    "neighbors": {
      "north": { "id": "grid_4_4", "tags": ["industrial", "power_plant"] },
      "west": { "id": "grid_3_5", "tags": ["slums"] }
    }
  },
  "constraints": {
    "hard": ["must_have_power_access"],
    "soft": ["vibe_decay", "high_crime"]
  },
  "validation_rules": {
    "zone_id": { "type": "string", "pattern": "^[a-z_]+$", "max_length": 32 },
    "hazard_ids": { "type": "array", "items": { "pattern": "^[a-z_]+$" } },
    "tags": { "type": "array", "items": { "pattern": "^[A-Z][a-z]+(-[A-Z][a-z]+)*$" } }
  }
}
```

### 9.2 Solver Proposal (LLM -> Engine)

```json
{
  "request_id": "req_8821a",
  "proposal": {
    "zone_id": "checkpoint",
    "visual_description": "A rusted security gate checking for contamination.",
    "hazards": ["toxin"],
    "tags": ["military", "decay"],
    "reasoning": "Checkpoint fits between Slums and Power Plant; decay fits weather."
  }
}
```

### 9.3 Validator Response (Engine Internal)

```json
{
  "status": "PASS",
  "validated_payload": {
    "zone_id": "checkpoint",
    "components": {
      "Zone": "checkpoint",
      "HazardVolume": "toxin",
      "Visuals": { "skybox": "acid_green", "props": ["barricade"] }
    }
  }
}
```
