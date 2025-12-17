# Second Opinion MCP

A Model Context Protocol (MCP) server that provides a "Second Opinion" tool. It uses Google's Gemini Flash model to sanity-check AI response plans against the user's original intent, ensuring alignment and detecting potential issues before code is written.

## Features

-   **Alignment Check**: objectively evaluates if an AI's proposed plan matches the user's prompt.
-   **Context Awareness**: Uses a local vector store (RAG) to remember past interactions and assessments.
-   **Psychiatrist's Ledger**: Generates a private behavioral assessment of the AI (stored locally) and a public, constructive opinion (returned to the agent).
-   **Global Instructions**: Can incorporate a global instruction file to ensure the second opinion respects the agent's core directives.

## Configuration

This MCP server requires a Google Gemini API key.

### Environment Variables

-   `GEMINI_API_KEY`: Your Google Gemini API Key.
-   `GEMINI_MODEL`: (Optional) Model to use (default: `gemini-1.5-flash`).
-   `GLOBAL_INSTRUCTIONS_PATH`: (Optional) Path to a markdown file containing global instructions for the AI agent.

### MCP Config Example

```json
{
  "mcpServers": {
    "second-opinion": {
      "command": "node",
      "args": [
        "/path/to/second-opinion-mcp/index.js"
      ],
      "env": {
        "GEMINI_API_KEY": "YOUR_KEY_HERE",
        "GLOBAL_INSTRUCTIONS_PATH": "/path/to/global_instructions.md"
      }
    }
  }
}
```

## Usage

 The tool `get-second-opinion` takes three arguments:
1.  `user_prompt`: The original request from the user.
2.  `ai_response_plan`: The plan the AI is proposing.
3.  `context`: A description of the current situation or what led up to this point (used for pattern recognition).

## Mechanics

1.  **Input**: The tool receives the prompt, plan, and context.
2.  **Retrieval**: It searches the local `vector_store.json` for similar past contexts to see if the AI is repeating mistakes.
3.  **Analysis**: Gemini analyzes the plan against the user prompt, global instructions, and past history.
4.  **Output**:
    -   **Public Response**: A constructive critique returned to the calling agent.
    -   **Private Assessment**: A blunt behavioral analysis stored in the local "ledger" to inform future critiques.
