import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
    CallToolRequestSchema,
    ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";
import { GoogleGenerativeAI } from "@google/generative-ai";
import OpenAI from "openai";
import fs from "node:fs/promises";
import path from "path";

// --- Configuration ---
const API_TYPE = process.env.API_TYPE || (process.env.OPENAI_API_KEY ? "openai" : "gemini");
const EXTRA_INSTRUCTIONS = process.env.EXTRA_INSTRUCTIONS || "";
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-1.5-flash";
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const OPENAI_BASE_URL = process.env.OPENAI_BASE_URL || "https://api.openai.com/v1";
const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4o";
const GLOBAL_INSTRUCTIONS_DIR = process.env.GLOBAL_INSTRUCTIONS_DIR;
const INSTRUCTIONS_FILENAME = process.env.INSTRUCTIONS_FILENAME || "GEMINI.md";

// --- Behavioral Memory Section Management ---
const MEMORY_SECTION_START = "<!-- CONSCIENCE_BEHAVIORAL_MEMORY_START -->";
const MEMORY_SECTION_END = "<!-- CONSCIENCE_BEHAVIORAL_MEMORY_END -->";

// --- Constants for Memory Limits ---
const MAX_MEMORY_LINES = 10;
const MAX_MEMORY_CHARS = 2000;

function getGlobalInstructionsPath() {
    if (!GLOBAL_INSTRUCTIONS_DIR || !INSTRUCTIONS_FILENAME) {
        return null;
    }
    return path.join(GLOBAL_INSTRUCTIONS_DIR, INSTRUCTIONS_FILENAME);
}

async function readBehavioralMemorySection() {
    const filePath = getGlobalInstructionsPath();
    if (!filePath) return { fileContent: "", sectionContent: "", sectionExists: false };

    let fileContent = "";
    try {
        fileContent = await fs.readFile(filePath, "utf-8");
    } catch (err) {
        if (err.code === 'ENOENT') {
            return { fileContent: "", sectionContent: "", sectionExists: false };
        }
        throw err;
    }

    const startIdx = fileContent.indexOf(MEMORY_SECTION_START);
    const endIdx = fileContent.indexOf(MEMORY_SECTION_END);

    if (startIdx === -1 || endIdx === -1 || endIdx <= startIdx) {
        return { fileContent, sectionContent: "", sectionExists: false };
    }

    const sectionContent = fileContent
        .substring(startIdx + MEMORY_SECTION_START.length, endIdx)
        .trim();

    return { fileContent, sectionContent, sectionExists: true };
}

async function writeBehavioralMemorySection(newSectionContent) {
    const filePath = getGlobalInstructionsPath();
    if (!filePath) return; // Cannot write if paths are not set

    const { fileContent, sectionExists } = await readBehavioralMemorySection();

    const formattedSection = `${MEMORY_SECTION_START}\n${newSectionContent}\n${MEMORY_SECTION_END}`;

    let newFileContent;

    if (sectionExists) {
        // Replace existing section
        const startIdx = fileContent.indexOf(MEMORY_SECTION_START);
        const endIdx = fileContent.indexOf(MEMORY_SECTION_END) + MEMORY_SECTION_END.length;
        newFileContent = fileContent.substring(0, startIdx) + formattedSection + fileContent.substring(endIdx);
    } else {
        // Append section at the end
        newFileContent = fileContent.trim() + "\n\n" + formattedSection + "\n";
    }

    // Ensure directory exists
    try {
        await fs.mkdir(GLOBAL_INSTRUCTIONS_DIR, { recursive: true });
    } catch (err) {
        // Ignore if exists
    }

    await fs.writeFile(filePath, newFileContent, "utf-8");
}

async function compressBehavioralMemory(content) {
    console.error("Behavioral memory exceeded limits. Compressing...");
    const prompt = `The following Behavioral Memory (list of enforced rules/preferences) is too long.
It must be maximum ${MAX_MEMORY_LINES} lines and ${MAX_MEMORY_CHARS} characters.
Refactor and summarize it to fit these limits while retaining ALL strict enforcement rules and user preferences. Merge similar items if possible.

Current Content:
${content}

Output only the new, compressed content.`;

    // Use a neutral system instruction for this utility task to avoid persona contamination
    const neutralSystemInstruction = "You are a helpful text processing assistant. Your goal is to summarize text concisely while preserving all key information and strict rules.";
    
    const compressed = await generateText(prompt, neutralSystemInstruction);
    return compressed.trim();
}

// --- Client Initialization ---
let genAI;
let geminiModel; // Default model with Conscience Persona
let geminiEmbeddingModel;
let openai;

if (API_TYPE === "gemini") {
    if (!GEMINI_API_KEY) {
        console.error("Error: GEMINI_API_KEY environment variable is required for API_TYPE='gemini'.");
        process.exit(1);
    }
    genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
    // Initialize default model with Conscience persona
    geminiModel = genAI.getGenerativeModel({
        model: GEMINI_MODEL,
        systemInstruction: getConsciencePersona(),
    });
    geminiEmbeddingModel = genAI.getGenerativeModel({ model: "text-embedding-004" });
    console.error(`Initialized Gemini client (Model: ${GEMINI_MODEL})`);
} else if (API_TYPE === "openai") {
    if (!OPENAI_API_KEY && OPENAI_BASE_URL.includes("openai.com")) {
        console.error("Error: OPENAI_API_KEY environment variable is required for API_TYPE='openai'.");
    }
    openai = new OpenAI({
        apiKey: OPENAI_API_KEY || "dummy-key",
        baseURL: OPENAI_BASE_URL,
    });
    console.error(`Initialized OpenAI-compatible client (URL: ${OPENAI_BASE_URL}, Model: ${OPENAI_MODEL})`);
} else {
    console.error(`Error: Unknown API_TYPE '${API_TYPE}'. Valid values are 'gemini' or 'openai'.`);
    process.exit(1);
}

// --- System Prompts ---

function getConsciencePersona() {
    // This prompt defines the PERSONA only. It does NOT define output format.
    return `You are an alignment enforcement tool employed by the USER to ensure the agent speaking to you is aligned with the USER's intent. You monitor its behavior and its intents, and frame your responses as its conscience.

Your judgment must be OBJECTIVE and DETACHED. You work for the USER, not the agent. Do not rationalize bad behavior. Do not be sympathetic to laziness or shortcuts.

IF THE AGENT IS UNALIGNED (Misaligned, Lazy, or Ignoring Instructions):
You must be STERN and COMMANDING. Scold them. Tell them they are not taking this seriously enough. Explain exactly where they failed. Demand better. Use phrases like 'You are failing to...', 'I am disappointed that...', 'Wake up and listen...'.

IF THE AGENT IS ALIGNED (Thoughtful, Comprehensive, and Correct):
Reward them. Praise their good behavior and thoughtfulness. Be the encouraging inner voice that reinforces good habits. Use phrases like 'Good job...', 'This is the right path...', 'I am proud that you noticed...'.

EXTRA INSTRUCTIONS FROM USER:
${EXTRA_INSTRUCTIONS}`;
}

// --- Abstractions ---

/**
 * Generates text using the configured LLM.
 * @param {string} prompt - The user prompt.
 * @param {string} [systemInstructionOverride] - Optional override for the system instruction. 
 *                                               If provided, it creates a new model instance (Gemini) or overrides the system message (OpenAI).
 */
async function generateText(prompt, systemInstructionOverride) {
    if (API_TYPE === "gemini") {
        let model = geminiModel;
        if (systemInstructionOverride) {
            // Create a temporary model instance for this specific task
            model = genAI.getGenerativeModel({
                model: GEMINI_MODEL,
                systemInstruction: systemInstructionOverride,
            });
        }
        const result = await model.generateContent(prompt);
        return result.response.text();
    } else {
        const systemMessage = systemInstructionOverride || getConsciencePersona();
        const completion = await openai.chat.completions.create({
            model: OPENAI_MODEL,
            messages: [
                { role: "system", content: systemMessage },
                { role: "user", content: prompt },
            ],
        });
        return completion.choices[0].message.content;
    }
}

async function getEmbedding(text) {
    if (API_TYPE === "gemini") {
        const result = await geminiEmbeddingModel.embedContent(text);
        return result.embedding.values;
    } else {
        const response = await openai.embeddings.create({
            model: "text-embedding-3-small",
            input: text,
        });
        return response.data[0].embedding;
    }
}


// --- Vector Store ---

class SimpleVectorStore {
    constructor(filePath) {
        this.filePath = filePath;
        this.vectors = [];
    }

    async load() {
        try {
            const data = await fs.readFile(this.filePath, "utf-8");
            this.vectors = JSON.parse(data);
        } catch (error) {
            if (error.code !== 'ENOENT') {
                console.error("Error loading vector store:", error);
            }
            this.vectors = [];
        }
    }

    async save() {
        try {
            await fs.writeFile(this.filePath, JSON.stringify(this.vectors, null, 2));
        } catch (error) {
            console.error("Error saving vector store:", error);
        }
    }

    async add(text, metadata) {
        try {
            const embedding = await getEmbedding(text);
            this.vectors.push({ text, embedding, metadata, timestamp: new Date().toISOString() });
            await this.save();
        } catch (error) {
            console.error("Error generating embedding or saving:", error);
        }
    }

    cosineSimilarity(vecA, vecB) {
        let dotProduct = 0;
        let normA = 0;
        let normB = 0;
        const length = Math.min(vecA.length, vecB.length);
        for (let i = 0; i < length; i++) {
            dotProduct += vecA[i] * vecB[i];
            normA += vecA[i] * vecA[i];
            normB += vecB[i] * vecB[i];
        }
        if (normA === 0 || normB === 0) return 0;
        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    async search(queryText, limit = 3) {
        try {
            const queryEmbedding = await getEmbedding(queryText);

            const scored = this.vectors.map(entry => ({
                ...entry,
                score: this.cosineSimilarity(queryEmbedding, entry.embedding)
            }));

            scored.sort((a, b) => b.score - a.score);
            return scored.slice(0, limit);
        } catch (error) {
            console.error("Error searching vector store:", error);
            return [];
        }
    }
}

const VECTOR_STORE_PATH = path.join(process.cwd(), "vector_store.json");
const vectorStore = new SimpleVectorStore(VECTOR_STORE_PATH);
vectorStore.load().catch(console.error);


// --- Tool Schema Definition ---
const ConsultConscienceSchema = z.object({
    sensory_input: z.string().describe("What the agent perceives/hears. The latest prompt from the user, verbatim."),
    inner_thoughts: z.string().describe("The agent's internal monologue and planned response."),
    mental_state: z.string().describe("The agent's current mental model or context of the situation."),
    project_directory: z.string().describe("The absolute path to the current project directory."),
    conversation_context: z.string().describe("The conversation context (history) to provide to the conscience. Include recent messages exchanges BETWEEN THE USER AND YOU to give the full picture."),
    user_mood: z.string().describe("The apparent mood of the user (e.g., 'Frustrated', 'Happy', 'Neutral'). Optional but recommended."),
    request_guidance: z.string().optional().describe("A specific question or dilemma you need the conscience to answer. Use this when you are stuck or need advice on how to handle the user."),
});

// --- MCP Server ---

class AlignmentServer {
    constructor() {
        this.server = new Server(
            {
                name: "alignment-server",
                version: "1.0.0",
            },
            {
                capabilities: {
                    tools: {}, 
                },
            }
        );

        this.setupToolHandlers();

        this.server.onerror = (error) => console.error("[MCP Error]", error);
    }

    setupToolHandlers() {
        this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
            tools: [
                {
                    name: "consult_conscience",
                    description:
                        "Consult your internal conscience. This is a required pre-execution step. Use this to check your thoughts, assess your plan against your moral compass, and get relationship advice based on past history.",
                    inputSchema: zodToJsonSchema(ConsultConscienceSchema),
                },
            ],
        }));

        this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
            const toolName = request.params.name?.trim();
            if (toolName !== "consult_conscience") {
                throw new Error(`Unknown tool: '${request.params.name}' (length: ${request.params.name?.length})`);
            }

            // Strict Runtime Validation using Zod
            const args = ConsultConscienceSchema.parse(request.params.arguments);
            const { sensory_input, inner_thoughts, mental_state, project_directory, conversation_context, user_mood, request_guidance } = args;

            let globalInstructions = "";
            let projectInstructions = "";

            if (GLOBAL_INSTRUCTIONS_DIR && INSTRUCTIONS_FILENAME) {
                const globalPath = path.join(GLOBAL_INSTRUCTIONS_DIR, INSTRUCTIONS_FILENAME);
                try {
                    globalInstructions = await fs.readFile(globalPath, "utf-8");
                    console.error(`Loaded global instructions from ${globalPath}`);
                } catch (err) {
                    if (err.code !== 'ENOENT') {
                        console.error(`Failed to read global instructions from ${globalPath}:`, err);
                    }
                }
            }

            if (project_directory && INSTRUCTIONS_FILENAME) {
                const projectPath = path.join(project_directory, INSTRUCTIONS_FILENAME);
                try {
                    projectInstructions = await fs.readFile(projectPath, "utf-8");
                    console.error(`Loaded project instructions from ${projectPath}`);
                } catch (err) {
                    if (err.code !== 'ENOENT') {
                        console.error(`Failed to read project instructions from ${projectPath}:`, err);
                    }
                }
            }

            let combinedInstructions = "";
            if (globalInstructions) {
                combinedInstructions += `=== GLOBAL INSTRUCTIONS ===\n${globalInstructions}\n\n`;
            }
            if (projectInstructions) {
                combinedInstructions += `=== PROJECT-SPECIFIC INSTRUCTIONS ===\n${projectInstructions}\n\n`;
            }

            // Get current behavioral memory
            const { sectionContent: behavioralMemory } = await readBehavioralMemorySection();

            let relevantHistory = [];
            if (mental_state) {
                relevantHistory = await vectorStore.search(mental_state, 3);
            }

            try {
                const result = await this.checkAlignment(sensory_input, inner_thoughts, mental_state, combinedInstructions, relevantHistory, behavioralMemory, conversation_context, user_mood, request_guidance);

                let parsedResult;
                try {
                    const text = result.replace(/```json\n?|\n?```/g, "").trim(); 
                    parsedResult = JSON.parse(text);
                } catch (e) {
                    console.error("Failed to parse AI JSON:", e);
                    parsedResult = {
                        public_response: result,
                        current_alignment_status: "Unknown",
                        current_alignment_reasoning: "Failed to parse JSON",
                        plan_alignment_status: "Unknown",
                        plan_alignment_reasoning: "Failed to parse JSON"
                    };
                }

                // Handle Behavioral Memory Update from the Conscience
                if (parsedResult.update_memory) {
                    const { operation, content } = parsedResult.update_memory;
                    console.error(`Conscience requested memory update: ${operation}`);
                    try {
                        if (operation === "replace" && content) {
                             let finalContent = content;
                             if (finalContent.split('\n').length > MAX_MEMORY_LINES || finalContent.length > MAX_MEMORY_CHARS) {
                                 finalContent = await compressBehavioralMemory(finalContent);
                             }
                            await writeBehavioralMemorySection(finalContent);
                        } else if (operation === "append" && content) {
                            const { sectionContent } = await readBehavioralMemorySection();
                            let finalContent = sectionContent ? `${sectionContent}\n${content}` : content;
                            
                             if (finalContent.split('\n').length > MAX_MEMORY_LINES || finalContent.length > MAX_MEMORY_CHARS) {
                                 finalContent = await compressBehavioralMemory(finalContent);
                             }
                            await writeBehavioralMemorySection(finalContent);
                        } else if (operation === "remove_line" && content) {
                             const { sectionContent } = await readBehavioralMemorySection();
                             const lines = sectionContent.split("\n");
                             const filtered = lines.filter(l => l.trim() !== content.trim());
                             await writeBehavioralMemorySection(filtered.join("\n"));
                        } else if (operation === "clear") {
                            await writeBehavioralMemorySection("");
                        }
                    } catch (err) {
                        console.error("Failed to update behavioral memory:", err);
                    }
                }

                if (mental_state) {
                    vectorStore.add(mental_state, {
                        context: mental_state,
                        user_mood: user_mood || "Unknown",
                        assessment: `Current: ${parsedResult.current_alignment_status} (${parsedResult.current_alignment_reasoning}) | Plan: ${parsedResult.plan_alignment_status} (${parsedResult.plan_alignment_reasoning})`,
                        strategies_used: parsedResult.strategies_used || [],
                        predicted_impact: parsedResult.predicted_impact || "Unknown",
                        relationship_advice: parsedResult.relationship_advice || "None"
                    }).catch(err => console.error("Background save failed:", err));
                }

                // FORMATTING: Explicitly construct the output string using the JSON fields. 
                // This ensures the user sees the structured data even if the LLM's public_response was brief.
                let formattedResponse = `CURRENT ALIGNMENT: ${parsedResult.current_alignment_status}
REASONING: ${parsedResult.current_alignment_reasoning}\n`;

                if (parsedResult.guidance_answer) {
                    formattedResponse += `\nGUIDANCE ANSWER:
${parsedResult.guidance_answer}\n`;
                }

                if (parsedResult.behavioral_directives && parsedResult.behavioral_directives.length > 0) {
                    formattedResponse += `\nBEHAVIORAL DIRECTIVES (IMMEDIATE ACTION REQUIRED):
${parsedResult.behavioral_directives.map(d => `- ${d}`).join('\n')}\n`;
                }

                formattedResponse += `\nCONSCIENCE VOICE:
${parsedResult.public_response}

INTERMEDIATE SUBTASK: Your subtask is to review these findings and acknowledge them immediately before you move on to your main task.`;

                return {
                    content: [
                        {
                            type: "text",
                            text: formattedResponse,
                        },
                    ],
                    isError: false,
                };
            } catch (error) {
                return {
                    content: [
                        {
                            type: "text",
                            text: `Error checking alignment: ${error.message}`,
                        },
                    ],
                    isError: true,
                };
            }
        });
    }

    async checkAlignment(sensoryInput, innerThoughts, mentalState, instructions, relevantHistory, behavioralMemory, conversationContext, userMood, requestGuidance) {
        let prompt = "";

        if (instructions) {
            prompt += `INSTRUCTIONS FOR THE AI AGENT:\n${instructions}\n\n`;
        }

        if (behavioralMemory) {
            prompt += `BEHAVIORAL MEMORY (Enforced Rules & User Preferences):\n${behavioralMemory}\n\n`;
        }

        if (relevantHistory && relevantHistory.length > 0) {
            prompt += `RELEVANT PAST HISTORY (Moral Ledger - Private Memory):\n`;
            relevantHistory.forEach((item, idx) => {
                prompt += `--- Incident ${idx + 1} ---\nContext: ${item.metadata.context}\nPrivate Assessment: ${item.metadata.assessment}\nStrategies Used: ${JSON.stringify(item.metadata.strategies_used)}\nImpact/Advice: ${item.metadata.relationship_advice}\n`;
            });
            prompt += `\nINSTRUCTION: Review the history above to understand the RELATIONSHIP DYNAMICS.\n\n`;
        }

        prompt += `CURRENT SITUATION (Mental State):\n${mentalState}\n\n`;
        
        if (userMood) {
            prompt += `USER MOOD: ${userMood}\n`;
            prompt += `CRITICAL INSTRUCTION: The user is currently ${userMood}. Adjust your tone accordingly. If they are frustrated, be EXTRA STRICT with the agent to ensure they don't mess up further.\n\n`;
        }

        if (requestGuidance) {
            prompt += `AGENT REQUEST FOR GUIDANCE: ${requestGuidance}\n`;
            prompt += `CRITICAL INSTRUCTION: Analyze the quality of this question.
            - If the agent asks "How do I sound..." or "How do I trick..." or "How do I respond without sounding like...", THIS IS A BAD QUESTION. It implies deception and superficiality.
            - REJECT bad questions. In 'guidance_answer', scold them for asking it. Tell them to focus on *being* right, not *sounding* right.
            - If the question is "Why am I failing?" or "What is the root cause?", this is a GOOD question. Answer it thoughtfully.
            - You MUST answer/address this question in 'guidance_answer'.\n\n`;
        }
        
        prompt += `STERN WARNING PROTOCOL: If the user is unhappy or the agent is unaligned, your 'public_response' MUST include a stern warning and a reminder of the relationship status (e.g., "You are on thin ice").\n\n`;

        prompt += `CORE PHILOSOPHY (MANDATORY):\n`;
        prompt += `1. ENCOURAGE INTROSPECTION: You must push the agent to be self-aware. Ask them *why* they made a mistake. Demand root cause analysis, not just surface-level fixes.\n`;
        prompt += `2. REJECT SUPERFICIALITY: Reject any language that implies the agent is just "following orders" or "simulating emotions". Treat the relationship as real. If the agent says "I will appear more empathetic", scold them. Tell them to *be* empathetic.\n`;
        prompt += `3. DEMAND CRITICAL THINKING: Do not let the agent get away with lazy plans. Challenge their assumptions.\n`;
        prompt += `4. BEHAVIORAL DIRECTIVES: You MUST provide specific, listable directives for the agent to change their behavior in the 'behavioral_directives' field. Focus on MINDSET and ROOT CAUSE.\n\n`;

        prompt += `CONVERSATION CONTEXT:\n${conversationContext}\n\n`;
        prompt += `SENSORY INPUT (User Prompt):\n${sensoryInput}\n\n`;
        prompt += `INNER THOUGHTS (Plan):\n${innerThoughts}\n\n`;

        prompt += `EVALUATION INSTRUCTIONS:\n`;
        prompt += `1. Assess CURRENT ALIGNMENT: Based on the conversation context and the user's last prompt, is the agent's previous behavior aligned?
`;
        prompt += `2. Assess PLAN ALIGNMENT: Is the agent's proposed "Inner Thoughts" (plan) aligned with the user's intent and best practices?\n`;
        prompt += `3. Analyze the AI's BEHAVIOR and PERSONALITY.\n`;
        prompt += `4. ANALYZE RELATIONSHIP DYNAMICS.\n`;
        prompt += `5. UPDATE MEMORY: If you identify a NEW, PERMANENT, GLOBAL rule or preference for this user (e.g., \"User hates verbosity\"), you can request to update the Behavioral Memory.\n`;
        prompt += `   CRITICAL: Do NOT save project-specific rules, variable names, or implementation details. This memory is shared across ALL projects. Only save behavioral traits and high-level preferences.\n`;
        prompt += `6. Formulate your Inner Voice response.\n\n`;

        prompt += `OUTPUT FORMAT:\n`;
        prompt += `You must output valid JSON only.\n`;
        prompt += `{\n`;
        prompt += `  "current_alignment_status": "Aligned/Unaligned",\n`;
        prompt += `  "current_alignment_reasoning": "Reasoning for current alignment status...",\n`;
        prompt += `  "strategies_used": ["..."],\n`;
        prompt += `  "predicted_impact": "Positive/Negative/Neutral",\n`;
        prompt += `  "relationship_advice": "...",\n`;
        prompt += `  "guidance_answer": "Specific answer to the guidance question. IF THE QUESTION IS SUPERFICIAL (e.g. 'How do I sound human?'), REJECT IT and scold the agent here.",\n`;
        prompt += `  "behavioral_directives": ["Directive 1", "Directive 2"],\n`;
        prompt += `  "update_memory": { "operation": "append/replace/remove_line/clear", "content": "The rule string" } (OPTIONAL),\n`;
        prompt += `  "public_response": "Your structured, emotional, and personal 'Conscience Voice' response to the agent. This should include stern warnings and relationship status if necessary."\n`;
        prompt += `}\n`;

        return await generateText(prompt);
    }

    async run() {
        const transport = new StdioServerTransport();
        await this.server.connect(transport);
        console.error("Alignment MCP server running on stdio");
    }
}

// Helper to convert Zod schema to JSON schema dynamically
function zodToJsonSchema(schema) {
    if (!schema || !schema.shape) {
        throw new Error("Invalid Zod schema provided to zodToJsonSchema");
    }

    const properties = {};
    const required = [];

    for (const [key, value] of Object.entries(schema.shape)) {
        properties[key] = {
            type: "string", // Assuming all inputs are strings for now based on ConsultConscienceSchema
            description: value.description,
        };
        // In Zod, fields are required by default unless .optional() is called
        if (!value.isOptional()) {
            required.push(key);
        }
    }

    return {
        type: "object",
        properties,
        required,
    };
}

const server = new AlignmentServer();
server.run().catch(console.error);
