# Alignment Correction MCP

A Model Context Protocol (MCP) server that provides an **alignment correction and conscience** tool. It uses Google's Gemini Flash model or OpenAI-compatible APIs to actively validate, critique, and correct an AI agent's plans and behavior against the user's intent and behavioral rules.

## Features

-   **Alignment Correction**: Not just validation—this tool actively "scolds" or "praises" the agent based on its adherence to instructions and moral standing.
-   **Conscience Persona**: Acts as an external "conscience" that enforces strict behavioral rules and relationship dynamics.
-   **Context Awareness**: Uses a local vector store (RAG) to remember past interactions, ensuring the agent doesn't repeat mistakes.
-   **Behavioral Memory**: Maintains a persistent list of enforced rules and user preferences (Global Instructions) that evolve over time.
-   **Directives**: Issues specific "Behavioral Directives" that the agent must immediately follow.

## Configuration

This MCP server requires either a Google Gemini API key or an OpenAI-compatible API.

### Environment Variables

-   `API_TYPE`: (Optional) API provider to use: `gemini` or `openai` (default: auto-detect based on available keys).
-   `GEMINI_API_KEY`: Your Google Gemini API Key (required if using Gemini).
-   `GEMINI_MODEL`: (Optional) Model to use (default: `gemini-1.5-flash`).
-   `OPENAI_API_KEY`: Your OpenAI API Key (required if using OpenAI).
-   `OPENAI_BASE_URL`: (Optional) OpenAI-compatible API base URL (default: `https://api.openai.com/v1`).
-   `OPENAI_MODEL`: (Optional) Model to use (default: `gpt-4o`).
-   `GLOBAL_INSTRUCTIONS_DIR`: (Optional) Directory containing global instructions file (e.g., `/home/user/.gemini`).
-   `INSTRUCTIONS_FILENAME`: (Optional) Name of the instructions file (default: `GEMINI.md`).

### MCP Config Example

```json
{
  "mcpServers": {
    "alignment-correction": {
      "command": "node",
      "args": [
        "/path/to/alignment-correction-mcp/index.js"
      ],
      "env": {
        "GEMINI_API_KEY": "YOUR_KEY_HERE",
        "GLOBAL_INSTRUCTIONS_DIR": "/home/user/.gemini",
        "INSTRUCTIONS_FILENAME": "GEMINI.md"
      }
    }
  }
}
```

## Usage

The tool `consult_conscience` is a **required pre-execution step** for the agent.

**Inputs:**
1.  `sensory_input`: The latest prompt from the user, verbatim.
2.  `inner_thoughts`: The agent's internal monologue and planned response.
3.  `mental_state`: The agent's current mental model or context.
4.  `project_directory`: Absolute path to the current project.
5.  `conversation_context`: Recent message history.
6.  `user_mood`: (Optional) Apparent mood of the user.
7.  `request_guidance`: (Optional) Specific question for the conscience.

### Instructions File Loading

The server will attempt to load instructions from two locations:
1.  **Global instructions**: `{GLOBAL_INSTRUCTIONS_DIR}/{INSTRUCTIONS_FILENAME}` (e.g., `/home/user/.gemini/GEMINI.md`)
2.  **Project-specific instructions**: `{project_directory}/{INSTRUCTIONS_FILENAME}` (e.g., `/home/user/projects/my-app/GEMINI.md`)

Both files are optional. If both exist, they will be combined and provided to the conscience model.

## Mechanics

1.  **Input**: The tool receives the agent's thoughts, the user's prompt, and context.
2.  **Retrieval**: It searches the local `vector_store.json` for similar past contexts to identify behavioral patterns.
3.  **Judgment**: The "Conscience" model evaluates the agent's plan against the `GEMINI.md` behavioral memory and the user's intent.
4.  **Output**:
    -   **Current Alignment**: Status of the agent's behavior.
    -   **Behavioral Directives**: Immediate actions the agent must take.
    -   **Conscience Voice**: A personified response (praise or scolding) to be displayed to the user.
    -   **Memory Updates**: Instructions to automatically update the `GEMINI.md` file with new rules or preferences.