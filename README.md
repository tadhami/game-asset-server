# game-asset-server

An MCP server that generates 2D game assets (rag-doll-style character turnarounds) by routing prompts to `google/gemini-3.1-flash-image-preview` via OpenRouter.

## Prerequisites

- Python 3.10 or newer
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`
- An [OpenRouter](https://openrouter.ai/) API key

## Installation

### Option A — uvx (recommended)

`uvx` runs the server in an ephemeral environment so you don't have to manage a venv. Point it at this checkout (or a published package once one exists) from your Claude Desktop config:

```json
{
  "mcpServers": {
    "game-asset-server": {
      "command": "uvx",
      "args": [
        "--from",
        "/Users/you/mcp-servers/game-asset-server",
        "game-asset-server"
      ],
      "env": {
        "OPENROUTER_API_KEY": "sk-or-v1-...",
        "GAME_ASSETS_DIR": "/Users/you/game-assets"
      }
    }
  }
}
```

### Option B — manual install

```bash
git clone <repo-url> game-asset-server
cd game-asset-server
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Then in `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "game-asset-server": {
      "command": "/path/to/game-asset-server/.venv/bin/game-asset-server",
      "env": {
        "OPENROUTER_API_KEY": "sk-or-v1-..."
      }
    }
  }
}
```

## Configuration

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENROUTER_API_KEY` | Yes | — | Your OpenRouter API key |
| `GAME_ASSETS_DIR` | No | `~/game-assets` | Where generated images are saved |
| `LOG_LEVEL` | No | `WARNING` | Set to `DEBUG` for verbose API-response logging |

The server fails fast at startup with a clear `ValueError` if `OPENROUTER_API_KEY` is missing.

## Tool reference

### `generate_2d_asset(prompt, reference_image_path?)`

Generates a 2D asset from a text prompt and writes the PNG to `GAME_ASSETS_DIR`.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `prompt` | string | Yes | Description of the character or asset to generate |
| `reference_image_path` | string | No | Path to a local image to use as a visual starting point. `~` is expanded. Must be an existing file with a `.png`, `.jpg`, `.jpeg`, `.webp`, or `.gif` extension. |

Returns an MCP `ImageContent` block (so Claude can see the result inline) plus a `TextContent` block with the saved file path.

## Model and style

The system prompt enforces a consistent rag-doll Metroidvania aesthetic — button eyes, mitten hands, flat colors, bold black outlines — and produces front/side/back turnaround sheets on a white background. Edit `SYSTEM_PROMPT` in `server.py` if you want a different look.

The underlying model is `google/gemini-3.1-flash-image-preview`, accessed through OpenRouter's OpenAI-compatible chat completions endpoint.
