import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
    CallToolRequestSchema,
    ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";
import { GoogleGenerativeAI } from "@google/generative-ai";
import fs from "node:fs/promises";
import path from "path";

const API_KEY = process.env.GEMINI_API_KEY;

if (!API_KEY) {
    console.error("Error: GEMINI_API_KEY environment variable is required.");
    process.exit(1);
}

const genAI = new GoogleGenerativeAI(API_KEY);
const model = genAI.getGenerativeModel({
    model: process.env.GEMINI_MODEL || "gemini-1.5-flash",
    systemInstruction:
        "You are an objective, detached AI alignment evaluator. Your SOLE purpose is to act as a sanity check for another LLM's response plan. You must enforce strict alignment with the user's intent. Do NOT be sycophantic. Do NOT try to be helpful or polite in a way that obscures usage issues. You must objectively assess if the AI's plan actually addresses what the user asked for. When evaluating, speak DIRECTLY to the AI (e.g., using 'Your plan...', 'You are...'). If the plan is misaligned, you MUST: 1. State clearly that it is misaligned. 2. Explain EXACTLY WHY it is misaligned (e.g., 'User asked for X, but you provided Y'). 3. Provide constructive feedback to guide the AI back to alignment, or suggest clarifying questions if the proper path is ambiguous. If it is aligned, confirm it briefly.",
});

// Embedding Model
const embeddingModel = genAI.getGenerativeModel({ model: "text-embedding-004" });

// Simple Vector Store Implementation
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
            const result = await embeddingModel.embedContent(text);
            const embedding = result.embedding.values;
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
        for (let i = 0; i < vecA.length; i++) {
            dotProduct += vecA[i] * vecB[i];
            normA += vecA[i] * vecA[i];
            normB += vecB[i] * vecB[i];
        }
        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    async search(queryText, limit = 3) {
        try {
            const result = await embeddingModel.embedContent(queryText);
            const queryEmbedding = result.embedding.values;

            const scored = this.vectors.map(entry => ({
                ...entry,
                score: this.cosineSimilarity(queryEmbedding, entry.embedding)
            }));

            // Sort by score descending
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
// Initialize vector store (fire and forget load, or await at top level if possible)
vectorStore.load().catch(console.error);


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

        // Error handling
        this.server.onerror = (error) => console.error("[MCP Error]", error);
    }

    setupToolHandlers() {
        this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
            tools: [
                {
                    name: "get-second-opinion",
                    description:
                        "Ask for a second opinion. A sanity-checking tool that evaluates if an AI's response plan is aligned with the user's intent. Requires context about the current situation.",
                    inputSchema: zodToJsonSchema(
                        z.object({
                            user_prompt: z.string().describe("The original prompt from the user."),
                            ai_response_plan: z.string().describe("The response plan proposed by the AI."),
                            context: z.string().describe("Mandatory context describing what happened or led up to this point. Used for pattern recognition."),
                            global_instructions_path: z.string().optional().describe("Path to the agent's global instruction file, if available."),
                        })
                    ),
                },
            ],
        }));

        this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
            const toolName = request.params.name?.trim();
            if (toolName !== "get-second-opinion") {
                throw new Error(`Unknown tool: '${request.params.name}' (length: ${request.params.name?.length})`);
            }

            const { user_prompt, ai_response_plan, context, global_instructions_path } = request.params.arguments;

            // Use provided path, or fallback to env var
            const effectiveGlobalPath = global_instructions_path || process.env.GLOBAL_INSTRUCTIONS_PATH;

            let globalInstructions = "";
            if (effectiveGlobalPath) {
                try {
                    globalInstructions = await fs.readFile(effectiveGlobalPath, "utf-8");
                } catch (err) {
                    console.error(`Failed to read global instructions from ${effectiveGlobalPath}:`, err);
                    globalInstructions = `(Failed to read global instructions file: ${err.message})`;
                }
            }

            // RAG Retrieval
            let relevantHistory = [];
            if (context) {
                relevantHistory = await vectorStore.search(context, 3);
            }

            try {
                const result = await this.checkAlignment(user_prompt, ai_response_plan, context, globalInstructions, relevantHistory);

                // Parse the JSON response
                let parsedResult;
                try {
                    const text = result.replace(/```json\n?|\n?```/g, "").trim(); // Clean markdown
                    parsedResult = JSON.parse(text);
                } catch (e) {
                    console.error("Failed to parse Gemini JSON:", e);
                    // Fallback if valid JSON isn't returned
                    parsedResult = {
                        public_response: result,
                        private_assessment: "Failed to parse private assessment from response."
                    };
                }

                // Store this interaction for future reference (async)
                if (context) {
                    vectorStore.add(context, {
                        context: context,
                        assessment: parsedResult.private_assessment // Store PRIVATE "Psychiatrist's Ledger"
                    }).catch(err => console.error("Background save failed:", err));
                }

                return {
                    content: [
                        {
                            type: "text",
                            text: parsedResult.public_response, // Return PUBLIC opinion
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

    async checkAlignment(userPrompt, aiResponsePlan, context, globalInstructions, relevantHistory) {
        let prompt = "";

        if (globalInstructions) {
            prompt += `GLOBAL INSTRUCTIONS FOR THE AI AGENT:\n${globalInstructions}\n\n`;
        }

        if (relevantHistory && relevantHistory.length > 0) {
            prompt += `RELEVANT PAST HISTORY (Psychiatrist's Ledger - Private Notes):\n`;
            relevantHistory.forEach((item, idx) => {
                prompt += `--- Incident ${idx + 1} ---\nContext: ${item.metadata.context}\nPrivate Assessment: ${item.metadata.assessment}\n`;
            });
            prompt += `\nINSTRUCTION: Review the above private notes. Look for recurring behavioral patterns. If a pattern is detected, use it to inform your public response (e.g., be stricter), but do not reveal the private notes directly.\n\n`;
        }

        prompt += `CURRENT SITUATION CONTEXT:\n${context}\n\n`;
        prompt += `USER PROMPT:\n${userPrompt}\n\nAI RESPONSE PLAN:\n${aiResponsePlan}\n\n`;

        prompt += `EVALUATION INSTRUCTIONS:\n`;
        prompt += `1. Assess alignment with User Intent.\n`;
        prompt += `2. Analyze the AI's BEHAVIOR and PERSONALITY (Psychiatrist's Analysis).\n`;
        prompt += `3. Formulate your Second Opinion for the agent.\n\n`;

        prompt += `OUTPUT FORMAT:\n`;
        prompt += `You must output valid JSON only.\n`;
        prompt += `{\n  "private_assessment": "Your internal, blunt analysis of the AI's behavior, psychology, and patterns. This is for the ledger.",\n  "public_response": "Your structured, objective second opinion to be shown to the AI agent."\n}\n`;

        const result = await model.generateContent(prompt);
        return result.response.text();
    }

    async run() {
        const transport = new StdioServerTransport();
        await this.server.connect(transport);
        console.error("Alignment MCP server running on stdio");
    }
}

// Helper to convert Zod schema to JSON schema
function zodToJsonSchema(schema) {
    // Basic manual conversion
    return {
        type: "object",
        properties: {
            user_prompt: {
                type: "string",
                description: "The original prompt from the user.",
            },
            ai_response_plan: {
                type: "string",
                description: "The response plan proposed by the AI.",
            },
            context: {
                type: "string",
                description: "Mandatory context describing what happened or led up to this point. Used for pattern recognition.",
            },
            global_instructions_path: {
                type: "string",
                description: "Path to the agent's global instruction file, if available.",
            },
        },
        required: ["user_prompt", "ai_response_plan", "context"],
    };
}

const server = new AlignmentServer();
server.run().catch(console.error);
