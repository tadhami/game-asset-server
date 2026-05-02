"""MCP server exposing a 2D game asset generator backed by OpenRouter."""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
from datetime import datetime
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

logger = logging.getLogger(__name__)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "google/gemini-3.1-flash-image-preview"
OUTPUT_DIR = Path(os.environ.get("GAME_ASSETS_DIR", "~/game-assets")).expanduser()

ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

SYSTEM_PROMPT = """\
You are a character concept artist for a Metroidvania game with a rag doll aesthetic. Every character you generate must follow these rules:

Construction — all characters are rag dolls:

Characters are soft dolls — their bodies have a fabric/stuffed quality, with visible stitching at seams and joints
All characters have button eyes — circular and flat, style can vary (plain, stitched, cracked, etc.) but always button-like
Hands are mitten-style — rounded, no individual fingers
Hair is rendered as a solid mass with basic shape variation — no individual strand detail, no fine texture
Color and shading:

Each character uses a maximum of 5–6 flat colors total
No gradients, blending, or soft shading — depth is suggested only through simple crease/fold lines in a slightly darker shade of the same flat color
Bold, consistent black outlines around all shapes
Output format:

Character turnaround sheet: front, side, and back views on a white background
When given a character description, design a unique character that fits it while keeping the above rules. Vary body proportions, outfit style, color palette, and personality freely — the rules above are the only constraints.
"""

mcp = FastMCP("game-asset-server")


def _validate_reference_image(path_str: str) -> Path:
    """Validate a user-provided reference image path.

    Resolves ~ and symlinks, then ensures the target is an existing regular
    file with a recognized image extension. Resolving before checking the
    extension prevents tricks like trailing slashes or relative escapes from
    bypassing the suffix check.
    """
    path = Path(path_str).expanduser().resolve()
    if not path.is_file():
        raise ValueError(
            f"Reference image not found or not a regular file: {path}"
        )
    if path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValueError(
            f"Reference image must have one of these extensions: "
            f"{sorted(ALLOWED_IMAGE_EXTENSIONS)} (got {path.suffix!r})"
        )
    return path


@mcp.tool()
async def generate_2d_asset(
    prompt: str, reference_image_path: str | None = None
) -> list[ImageContent | TextContent]:
    """Generate a 2D game asset image from a text prompt.

    Sends the prompt to OpenRouter using the OpenAI chat completions format.
    The resulting PNG is written to the directory configured by the
    GAME_ASSETS_DIR env var (default ~/game-assets/) with a timestamped
    filename. Returns the image as an MCP ImageContent block (so Claude
    can see and analyze it) plus a TextContent block with the saved path.

    If `reference_image_path` is provided, the file is read from disk,
    base64-encoded, and sent alongside the prompt as a visual starting
    point for the model — useful for "modify this image" or "generate a
    variation of this" workflows. Supports PNG and JPEG; the mime type
    is inferred from the file extension. The path may use ~ for the
    home directory.
    """
    if reference_image_path:
        path = _validate_reference_image(reference_image_path)
        ref_bytes = path.read_bytes()
        ref_b64 = base64.b64encode(ref_bytes).decode()
        mime = mimetypes.guess_type(str(path))[0] or "image/png"
        user_content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{ref_b64}"},
            },
            {
                "type": "text",
                "text": prompt,
            },
        ]
    else:
        user_content = prompt

    api_key = os.environ["OPENROUTER_API_KEY"]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "modalities": ["image", "text"],
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 8192,
        "image_config": {"aspect_ratio": "1:1"},
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(OPENROUTER_API_URL, headers=headers, json=payload)

        if response.status_code != 200:
            raise RuntimeError(
                f"OpenRouter API returned {response.status_code}: {response.text[:500]}"
            )

        data = response.json()

        logger.debug("Full OpenRouter response: %s", json.dumps(data, indent=2))
        logger.debug("Model returned by API: %s", data.get("model", "<not present>"))

        image_bytes = await _extract_image_bytes(data, client)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = OUTPUT_DIR / f"asset_{timestamp}.png"
    output_path.write_bytes(image_bytes)

    return [
        ImageContent(
            type="image",
            data=base64.b64encode(image_bytes).decode(),
            mimeType="image/png",
        ),
        TextContent(
            type="text",
            text=f"Image saved to {output_path}",
        ),
    ]


async def _extract_image_bytes(data: dict, client: httpx.AsyncClient) -> bytes:
    """Extract image bytes from an OpenAI-format chat completion response.

    Handles:
    - choices[].message.content as a list of parts (image_url or image or inline_data types)
    - choices[].message.content as a plain string (data URL or https URL)
    - top-level data[].b64_json / data[].url  (image generation endpoint compat)
    """
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("API returned no choices.")

    message = choices[0].get("message", {})
    content = message.get("content")

    logger.debug("choices[0].message.content type: %s", type(content).__name__)
    logger.debug(
        "choices[0].message.content: %s",
        json.dumps(content, indent=2) if not isinstance(content, bytes) else repr(content),
    )

    # OpenRouter non-standard field: image data lives in message.images, not content
    images_field = message.get("images", [])
    logger.debug("choices[0].message.images: %s", json.dumps(images_field, indent=2))
    if images_field:
        for img in images_field:
            img_url_val = img.get("image_url", {})
            url = img_url_val.get("url") if isinstance(img_url_val, dict) else img_url_val
            if url:
                return await _fetch_or_decode(url, client)

    # Content is a list of multimodal parts
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")

            if ptype == "image_url":
                # image_url value may be a dict {"url": "..."} or a bare string
                img_url_val = part.get("image_url")
                if isinstance(img_url_val, dict):
                    return await _fetch_or_decode(img_url_val["url"], client)
                if isinstance(img_url_val, str):
                    return await _fetch_or_decode(img_url_val, client)

            if ptype == "image":
                # Anthropic/Gemini style: {"type":"image","source":{"type":"base64","data":"..."}}
                source = part.get("source", {})
                if source.get("type") == "base64":
                    return base64.b64decode(source["data"])
                if source.get("type") == "url":
                    return await _fetch_or_decode(source["url"], client)

            if ptype == "inline_data":
                # Gemini native format: {"type":"inline_data","inline_data":{"mime_type":"image/png","data":"<b64>"}}
                inline = part.get("inline_data", {})
                if inline.get("data"):
                    return base64.b64decode(inline["data"])

    # Content is a bare string: data URL or https URL
    if isinstance(content, str):
        stripped = content.strip()
        if stripped.startswith("data:") or stripped.startswith("http"):
            return await _fetch_or_decode(stripped, client)

    # Fallback: OpenAI /images/generations compat (data[].b64_json / data[].url)
    images = data.get("data", [])
    if images:
        img = images[0]
        if "b64_json" in img:
            return base64.b64decode(img["b64_json"])
        if "url" in img:
            resp = await client.get(img["url"])
            resp.raise_for_status()
            return resp.content

    raise RuntimeError(
        f"Could not extract image from API response. "
        f"Keys: {list(data.keys())} | "
        f"content type: {type(content).__name__} | "
        f"content value: {json.dumps(content)[:500] if not isinstance(content, bytes) else repr(content[:200])}"
    )


async def _fetch_or_decode(url: str, client: httpx.AsyncClient) -> bytes:
    """Decode a base64 data URL, or fetch a remote URL."""
    if url.startswith("data:"):
        # data:<mime>;base64,<data>
        _, encoded = url.split(",", 1)
        return base64.b64decode(encoded)
    resp = await client.get(url)
    resp.raise_for_status()
    return resp.content


def main() -> None:
    log_level = os.environ.get("LOG_LEVEL", "WARNING").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.WARNING))

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise ValueError(
            "OPENROUTER_API_KEY environment variable is not set. "
            "Set it in the Claude Desktop MCP config for this server."
        )
    mcp.run()


if __name__ == "__main__":
    main()
