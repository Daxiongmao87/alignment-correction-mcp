# Alignment Validation MCP

A Model Context Protocol (MCP) server that provides an alignment validation tool. It uses Google's Gemini Flash model or OpenAI-compatible APIs to validate AI response plans against the user's original intent, ensuring alignment and detecting potential issues before code is written.

## Features

-   **Alignment Validation**: Objectively evaluates if an AI's proposed plan matches the user's prompt.
-   **Context Awareness**: Uses a local vector store (RAG) to remember past interactions and assessments.
-   **Behavioral Analysis**: Generates a private behavioral assessment of the AI (stored locally) and a public, constructive validation response (returned to the agent).
-   **Global Instructions**: Can incorporate a global instruction file to ensure the validation respects the agent's core directives.

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
    "alignment-validation": {
      "command": "node",
      "args": [
        "/path/to/alignment-validation-mcp/index.js"
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

The tool `validate-alignment` takes four required arguments:
1.  `user_prompt`: The original request from the user.
2.  `ai_response_plan`: The plan the AI is proposing.
3.  `context`: A description of the current situation or what led up to this point (used for pattern recognition).
4.  `project_directory`: The absolute path to the current project directory.

### Instructions File Loading

The server will attempt to load instructions from two locations:
1.  **Global instructions**: `{GLOBAL_INSTRUCTIONS_DIR}/{INSTRUCTIONS_FILENAME}` (e.g., `/home/user/.gemini/GEMINI.md`)
2.  **Project-specific instructions**: `{project_directory}/{INSTRUCTIONS_FILENAME}` (e.g., `/home/user/projects/my-app/GEMINI.md`)

Both files are optional. If both exist, they will be combined and provided to the validation model, with global instructions first, followed by project-specific instructions.

## Mechanics

1.  **Input**: The tool receives the prompt, plan, and context.
2.  **Retrieval**: It searches the local `vector_store.json` for similar past contexts to identify behavioral patterns.
3.  **Analysis**: The configured AI model analyzes the plan against the user prompt, global instructions, and past history.
4.  **Output**:
    -   **Public Response**: A validation assessment returned to the calling agent.
    -   **Private Assessment**: A behavioral analysis stored in the local vector store to inform future validations.
