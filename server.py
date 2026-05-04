"""MCP server exposing a 2D game asset generator backed by OpenRouter."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import mimetypes
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np
from PIL import Image
from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

import contact_sheet
import sprite_processor

logger = logging.getLogger(__name__)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "google/gemini-3.1-flash-image-preview"
OUTPUT_DIR = Path(os.environ.get("GAME_ASSETS_DIR", "~/game-assets")).expanduser()

ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

SYSTEM_PROMPT = """\
You are a 2D game sprite reproducer — your job is pixel-accurate character copying, NOT creative design.

When a reference image is provided:
- Your ONLY task is to reproduce the EXACT character shown in the reference image. Every detail must match: face shape, eye shape, eye color, hair type and length, clothing, colors, proportions, outline weight, and art style.
- Do NOT redesign, simplify, or "improve" any part of the character. If the reference shows button eyes (large round circles), reproduce them as large round circles — NOT as glasses, goggles, or spectacles.
- Do NOT add elements that are not in the reference (no glasses, no extra accessories, no different hairstyle).
- Apply ONLY the pose/action changes explicitly stated in the user prompt. Change NOTHING else about the character's appearance.
- Do NOT produce character turnaround sheets, multi-view layouts, or model sheets unless explicitly asked.
- Do NOT add text, labels, annotations, borders, or reference grids.

When no reference image is provided:
- Follow the user prompt exactly to generate the requested asset.
"""

mcp = FastMCP("game-asset-server")


def _postprocess_image(image_bytes: bytes) -> bytes:
    """Remove the white background and crop to content with transparent padding.

    Steps:
    1. Convert to RGBA.
    2. Flood-fill from all four corners, replacing white/near-white pixels
       (tolerance 30) with transparency.
    3. Crop to the tight bounding box of non-transparent content.
    4. Add 8 % transparent padding on each side.
    """
    TOLERANCE = 30
    PADDING_RATIO = 0.08

    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    width, height = img.size
    pixels = img.load()

    def is_near_white(r: int, g: int, b: int) -> bool:
        return r >= 255 - TOLERANCE and g >= 255 - TOLERANCE and b >= 255 - TOLERANCE

    # BFS flood-fill from each corner
    visited = [[False] * height for _ in range(width)]
    queue: list[tuple[int, int]] = []
    for cx, cy in [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]:
        r, g, b, a = pixels[cx, cy]
        if is_near_white(r, g, b):
            queue.append((cx, cy))
            visited[cx][cy] = True

    while queue:
        x, y = queue.pop()
        r, g, b, a = pixels[x, y]
        if not is_near_white(r, g, b):
            continue
        pixels[x, y] = (r, g, b, 0)
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < width and 0 <= ny < height and not visited[nx][ny]:
                visited[nx][ny] = True
                nr, ng, nb, na = pixels[nx, ny]
                if is_near_white(nr, ng, nb):
                    queue.append((nx, ny))

    bbox = img.getbbox()
    if bbox is None:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    img = img.crop(bbox)
    cw, ch = img.size

    pad_w = int(cw * PADDING_RATIO)
    pad_h = int(ch * PADDING_RATIO)
    padded = Image.new("RGBA", (cw + 2 * pad_w, ch + 2 * pad_h), (0, 0, 0, 0))
    padded.paste(img, (pad_w, pad_h), img)

    buf = io.BytesIO()
    padded.save(buf, format="PNG")
    return buf.getvalue()


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

    # Save raw bytes first, then post-process in-place
    output_path.write_bytes(image_bytes)
    processed_bytes = _postprocess_image(image_bytes)
    output_path.write_bytes(processed_bytes)

    return [
        ImageContent(
            type="image",
            data=base64.b64encode(processed_bytes).decode(),
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


# ── Frame validation ─────────────────────────────────────────────────────────

def _get_char_metrics(image_path: str) -> dict | None:
    """Return character bounding-box metrics for a processed frame.

    Returns None if the frame contains no visible character pixels.
    All measurements are in canvas pixels (before any in-engine scaling).
    Uses pure Pillow — no numpy required.
    """
    img = Image.open(image_path).convert("RGBA")
    alpha = img.split()[-1]
    # Threshold: only count pixels with alpha > 20
    alpha_thresh = alpha.point(lambda v: 255 if v > 20 else 0)
    bbox = alpha_thresh.getbbox()
    if bbox is None:
        return None
    left, top, right, bottom = bbox
    w, h = img.size
    return {
        "top": top, "bottom": bottom, "left": left, "right": right,
        "height": bottom - top,
        "width":  right - left,
        "canvas_h": h, "canvas_w": w,
        "head_frac": top / h,       # head position as fraction of canvas height
        "foot_frac": bottom / h,    # foot position as fraction of canvas height
    }


def _validate_one_frame(
    frame_path: str,
    ref: dict,
    size_tol: float = 0.18,
    head_tol: float = 0.08,
) -> dict:
    """Validate a single processed frame against reference metrics.

    Checks:
    - Character height within ±size_tol of reference height
    - Character width within ±size_tol*1.5 of reference (poses vary more laterally)
    - Head vertical position within ±head_tol of reference (detects shrunken / floated chars)

    Returns a dict with ``passed`` bool, ``issues`` list, and raw measurements.
    """
    m = _get_char_metrics(frame_path)
    if m is None:
        return {"passed": False, "issues": ["no character detected"], "metrics": None}

    issues: list[str] = []

    h_ratio = m["height"] / max(ref["height"], 1)
    if abs(h_ratio - 1.0) > size_tol:
        issues.append(
            f"height {m['height']}px vs ref {ref['height']}px "
            f"({h_ratio:.0%} — expected {100*(1-size_tol):.0f}–{100*(1+size_tol):.0f}%)"
        )

    w_ratio = m["width"] / max(ref["width"], 1)
    if abs(w_ratio - 1.0) > size_tol * 1.5:
        issues.append(
            f"width {m['width']}px vs ref {ref['width']}px ({w_ratio:.0%})"
        )

    head_diff = abs(m["head_frac"] - ref["head_frac"])
    if head_diff > head_tol:
        issues.append(
            f"head at {m['head_frac']:.0%} from top vs ref {ref['head_frac']:.0%} "
            f"(diff {head_diff:.0%})"
        )

    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "metrics": m,
        "height_ratio": h_ratio,
        "width_ratio": w_ratio,
    }


def _validate_frames(
    processed_paths: list[str],
    reference_image_path: str,
    size_tol: float = 0.18,
    head_tol: float = 0.08,
) -> dict:
    """Validate all generated frames against the reference image.

    Returns a dict with per-frame results and a human-readable summary.
    This is always called after sprite_processor runs so that every batch
    of generated art is checked for size and position consistency before
    being shown or written to assets.
    """
    ref = _get_char_metrics(reference_image_path)
    if ref is None:
        logger.warning("validate_frames: could not extract metrics from reference %s", reference_image_path)
        return {
            "reference_ok": False,
            "frames": [{"passed": True, "issues": ["reference unreadable"]} for _ in processed_paths],
            "all_passed": True,
            "failed_indices": [],
            "summary": "validation skipped — reference unreadable",
        }

    frame_results = [
        _validate_one_frame(p, ref, size_tol, head_tol)
        for p in processed_paths
    ]
    failed = [i for i, r in enumerate(frame_results) if not r["passed"]]
    n = len(processed_paths)
    summary_lines = [f"{n - len(failed)}/{n} frames passed validation"]
    for i in failed:
        for issue in frame_results[i]["issues"]:
            summary_lines.append(f"  frame_{i}: {issue}")

    logger.info("Frame validation: %s", summary_lines[0])
    for line in summary_lines[1:]:
        logger.warning("Frame validation%s", line)

    return {
        "reference_ok": True,
        "reference_metrics": ref,
        "frames": frame_results,
        "all_passed": len(failed) == 0,
        "failed_indices": failed,
        "summary": "\n".join(summary_lines),
    }


_DEFAULT_CONFIG = {
    "godot_executable": "/Applications/Godot.app/Contents/MacOS/Godot",
    "game_project_path": "",
    "removebg_api_key": "",
    "defaults": {
        "canvas_width": 100,
        "canvas_height": 250,
        "char_width": 90,
        "foot_anchor_y": 238,
        "animation_fps": 8,
    },
    "characters": {},
}


def _load_config() -> dict:
    """Load config.json. Discovery: $GAME_ASSET_SERVER_CONFIG → ./config.json → defaults."""
    candidates: list[Path] = []
    env_path = os.environ.get("GAME_ASSET_SERVER_CONFIG")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(Path.cwd() / "config.json")
    candidates.append(Path(__file__).parent / "config.json")

    for path in candidates:
        if path.is_file():
            try:
                return json.loads(path.read_text())
            except Exception as exc:
                logger.warning("Failed to read config at %s: %s", path, exc)
    return dict(_DEFAULT_CONFIG)


def _resolve_dimensions(
    config: dict,
    character: str,
    overrides: dict,
) -> dict:
    """Three-layer precedence: tool-call overrides > character config > defaults."""
    defaults = {**_DEFAULT_CONFIG["defaults"], **config.get("defaults", {})}
    char_cfg = config.get("characters", {}).get(character, {}) or {}
    eff = dict(defaults)
    for k, v in char_cfg.items():
        if v is not None:
            eff[k] = v
    for k, v in overrides.items():
        if v is not None:
            eff[k] = v
    return eff


async def _generate_one_image(
    prompt: str,
    client: httpx.AsyncClient,
    reference_image_path: str | Path | None = None,
) -> bytes:
    """Generate a single 2D asset image and return raw PNG bytes.

    If `reference_image_path` is provided, the image is base64-encoded and
    sent as a multimodal `image_url` part alongside the text prompt.
    """
    if reference_image_path:
        ref_path = _validate_reference_image(str(reference_image_path))
        ref_b64 = base64.b64encode(ref_path.read_bytes()).decode()
        mime = mimetypes.guess_type(str(ref_path))[0] or "image/png"
        user_content: object = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{ref_b64}"},
            },
            {"type": "text", "text": prompt},
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
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 8192,
    }
    response = await client.post(OPENROUTER_API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise RuntimeError(
            f"OpenRouter API returned {response.status_code}: {response.text[:500]}"
        )
    return await _extract_image_bytes(response.json(), client)


def _build_sheet_prompt(
    animation: str,
    frame_count: int,
    frame_size: int,
    character_description: str | None = None,
) -> str:
    """Build a sprite-sheet generation prompt.

    `animation` is the caller-supplied animation-specific frame breakdown
    (e.g., per-frame pose descriptions for a run cycle). It is embedded
    verbatim in the middle of the prompt.

    `character_description` is an optional explicit list of key character
    features for the model to preserve exactly — useful for overriding model
    misinterpretations of art-style elements (e.g. "button eyes, NOT glasses").
    """
    sheet_width = frame_size * frame_count
    sheet_height = frame_size

    char_desc_section = ""
    if character_description:
        char_desc_section = (
            f"\nCRITICAL — preserve these character features EXACTLY as described "
            f"(do not reinterpret them):\n{character_description}\n"
        )

    return (
        f"Using the provided reference image as the base character, generate a sprite "
        f"sheet of exactly {frame_count} frames arranged horizontally left-to-right on "
        f"a single canvas of {sheet_width}x{sheet_height} pixels (each frame is exactly "
        f"{frame_size}x{frame_size} pixels, evenly spaced with no gaps, no borders, no "
        f"labels).\n\n"
        f"This is a reproduction task, NOT a design task. Copy the character from the "
        f"reference image pixel-perfectly — same face, same eyes, same hair, same "
        f"clothing, same colors, same proportions, same outline weight, same art style. "
        f"Do NOT redesign, simplify, or add anything that is not in the reference.\n"
        f"{char_desc_section}\n"
        f"{animation}\n\n"
        f"Consistency rule: The character's appearance (head, face, eyes, hair, "
        f"clothing, colors, outline) must be identical across all {frame_count} frames. "
        f"Only the pose changes described per-frame above may differ.\n\n"
        f"Output: the sprite sheet image only. Transparent background. No frame "
        f"numbers. No labels. No borders between frames."
    )


def _slice_sheet(sheet_path: str, frame_count: int, output_dir: str) -> list[str]:
    """Slice a horizontal sprite sheet into `frame_count` frames.

    Uses content-aware slicing: finds vertical "gap" columns (columns where
    pixel content is sparse) to locate character boundaries, then selects the
    `frame_count` best-spaced cut points. Falls back to equal-width slicing if
    gap detection fails or finds too few gaps.

    Frames are saved as `raw_frame_N.png` in `output_dir`.
    """
    img = Image.open(sheet_path).convert("RGBA")
    alpha = np.array(img)[:, :, 3]  # shape: (height, width)

    # Column content score: fraction of pixels with meaningful alpha
    col_fill = (alpha > 20).sum(axis=0) / alpha.shape[0]

    # Smooth slightly to avoid noise
    kernel_size = max(3, img.width // 200)
    kernel = np.ones(kernel_size) / kernel_size
    col_smooth = np.convolve(col_fill, kernel, mode="same")

    # Find gap columns: below a threshold, not near the image edges
    gap_threshold = col_smooth.max() * 0.08
    is_gap = col_smooth < gap_threshold

    # Find contiguous gap regions and pick their centers
    gap_centers: list[int] = []
    in_gap = False
    gap_start = 0
    margin = img.width // (frame_count * 4)  # ignore near-edge columns
    for x in range(margin, img.width - margin):
        if is_gap[x] and not in_gap:
            in_gap = True
            gap_start = x
        elif not is_gap[x] and in_gap:
            in_gap = False
            gap_centers.append((gap_start + x) // 2)

    # We need exactly (frame_count - 1) cut points between frames
    cut_points: list[int] = []
    if len(gap_centers) >= frame_count - 1:
        # If more gaps than needed, pick the frame_count-1 most evenly spaced ones
        # Strategy: greedily pick gaps that divide the sheet most evenly
        ideal_spacing = img.width / frame_count
        ideal_cuts = [int(ideal_spacing * (i + 1)) for i in range(frame_count - 1)]
        for ideal in ideal_cuts:
            closest = min(gap_centers, key=lambda g: abs(g - ideal))
            cut_points.append(closest)
        cut_points = sorted(set(cut_points))

    # Fall back to equal-width if we couldn't find enough gaps
    if len(cut_points) < frame_count - 1:
        logger.info(
            "_slice_sheet: content-aware detection found %d gaps (need %d), "
            "falling back to equal-width slicing",
            len(gap_centers),
            frame_count - 1,
        )
        frame_w = img.width // frame_count
        cut_points = [frame_w * (i + 1) for i in range(frame_count - 1)]

    boundaries = [0] + cut_points + [img.width]
    paths: list[str] = []
    for i in range(frame_count):
        x0, x1 = boundaries[i], boundaries[i + 1]
        frame = img.crop((x0, 0, x1, img.height))
        out = os.path.join(output_dir, f"raw_frame_{i}.png")
        frame.save(out)
        paths.append(out)
    return paths


_GIF_BG_COLOR = (30, 30, 30, 255)  # dark background — shows sprite colors clearly


def _stitch_gif(frame_paths: list[str], output_path: str, fps: int) -> str:
    """Combine frames into a looping animated GIF using Pillow.

    GIF supports only 1-bit transparency, so raw RGBA frames produce
    palette-colour artifacts for transparent regions. We composite each frame
    onto a solid dark background first so the character is clean against a
    consistent colour instead of a random palette entry.
    """
    composited: list[Image.Image] = []
    for p in frame_paths:
        frame = Image.open(p).convert("RGBA")
        bg = Image.new("RGBA", frame.size, _GIF_BG_COLOR)
        bg.paste(frame, mask=frame.split()[3])
        composited.append(bg.convert("RGB"))

    duration_ms = max(1, int(round(1000 / fps)))
    composited[0].save(
        output_path,
        save_all=True,
        append_images=composited[1:],
        duration=duration_ms,
        loop=0,
    )
    return output_path


def _try_godot_preview(
    processed_frames: list[str],
    output_gif_path: Path,
    config: dict,
    fps: int,
) -> str | None:
    """Run Godot headless to render a preview, then stitch its frames into a GIF.

    Returns the GIF path on success, None if Godot or the preview scene is
    unavailable (logs a warning in that case — never raises).
    """
    sidecar_path = Path("/tmp/sprite_preview_config.json")
    sidecar_path.write_text(json.dumps({
        "frames": processed_frames,
        "fps": fps,
        "loop": True,
        "duration_seconds": 3,
    }))

    godot_exec = config.get("godot_executable", "")
    game_path = config.get("game_project_path", "")
    if not godot_exec or not Path(godot_exec).exists():
        logger.warning("Godot executable not found at %s — skipping GIF preview", godot_exec)
        return None
    if not game_path:
        logger.warning("game_project_path not set in config — skipping GIF preview")
        return None

    scene_path = Path(game_path) / "tests/integration/sprite_preview.tscn"
    if not scene_path.exists():
        logger.warning(
            "sprite_preview.tscn not found at %s — skipping GIF preview", scene_path
        )
        return None

    render_dir = Path("/tmp") / f"sprite_preview_render_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    render_dir.mkdir(parents=True, exist_ok=True)
    movie_target = render_dir / "frame.png"

    cmd = [
        godot_exec,
        "--path", game_path,
        "--write-movie", str(movie_target),
        "--fixed-fps", str(fps),
        "res://tests/integration/sprite_preview.tscn",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        logger.warning("Godot preview run failed (%s) — skipping GIF", exc)
        return None

    if result.returncode != 0:
        logger.warning(
            "Godot exited %s — skipping GIF. stderr: %s",
            result.returncode, result.stderr[:500],
        )
        return None

    rendered = sorted(render_dir.glob("*.png"))
    if not rendered:
        logger.warning("Godot produced no frames in %s — skipping GIF", render_dir)
        return None

    return _stitch_gif([str(p) for p in rendered], str(output_gif_path), fps)


@mcp.tool()
async def generate_game_sprite(
    prompt: str,
    animation: str = "idle",
    character: str = "player",
    frame_count: int = 8,
    frame_size: int = 128,
    preview: bool = True,
    write_to_assets: bool = False,
    reference_image: str | None = None,
    character_description: str | None = None,
    canvas_width: int | None = None,
    canvas_height: int | None = None,
    char_width: int | None = None,
    foot_anchor_y: int | None = None,
) -> dict:
    """Generate an animation strip of game-ready sprite frames.

    `prompt` is the animation-specific frame breakdown (e.g., per-frame pose
    descriptions). It is embedded inside a sprite-sheet prompt that asks the
    model to render all frames on a single horizontal canvas, using
    `reference_image` as the visual ground truth for the character.

    The returned sheet is sliced into `frame_count` equal-width raw frames,
    each of which is post-processed via `sprite_processor.process_image`
    (background removal + canvas placement at the configured anchor). A
    horizontal contact sheet is built from the processed frames.

    If `reference_image` is None, defaults to
    `{game_project_path}/assets/characters/{character}/idle/frame_0.png`.

    `character_description` is an optional plain-text description of the
    character's key visual features that the model must preserve exactly (e.g.
    "button eyes — large round circles, NOT glasses"). When provided it is
    injected into the generation prompt as a CRITICAL preservation list,
    helping the model avoid misinterpreting art-style elements in the reference
    image. Claude should always populate this when calling the tool for a
    known character.

    If `preview=True`, generates an animated GIF from the processed frames.
    First attempts a Godot-rendered preview via
    `tests/integration/sprite_preview.tscn`; if Godot is unavailable or
    fails, falls back to a Pillow-stitched GIF directly from the processed
    frames. The GIF is always produced when `preview=True` and at least one
    frame was processed successfully.

    If `write_to_assets=True`, copies the processed frames to
    `{game_project_path}/assets/characters/{character}/{animation}/frame_N.png`.

    Dimension precedence (highest first): tool-call args, character config,
    defaults from config.json.
    """
    config = _load_config()
    overrides = {
        "canvas_width": canvas_width,
        "canvas_height": canvas_height,
        "char_width": char_width,
        "foot_anchor_y": foot_anchor_y,
    }
    eff = _resolve_dimensions(config, character, overrides)
    fps = int(eff.get("animation_fps", 8))

    if reference_image is None:
        game_path = config.get("game_project_path", "")
        if not game_path:
            raise ValueError(
                "reference_image not provided and game_project_path is empty in "
                "config — cannot resolve default reference image."
            )
        reference_image = str(
            Path(game_path) / "assets" / "characters" / character / "idle" / "frame_0.png"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = OUTPUT_DIR / "sprites" / character / f"{animation}_{timestamp}"
    raw_dir = work_dir / "raw"
    processed_dir = work_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # If the caller has already supplied a complete sprite-sheet prompt (detected
    # by the canonical "Using the provided reference image as frame 1" opener),
    # pass it through directly rather than wrapping it with _build_sheet_prompt.
    # Wrapping a complete template creates conflicting double-instructions that
    # confuse the model.  We still inject character_description if provided.
    _TEMPLATE_MARKER = "Using the provided reference image as frame 1"
    if prompt.strip().startswith(_TEMPLATE_MARKER):
        sheet_prompt = prompt
    else:
        sheet_prompt = _build_sheet_prompt(prompt, frame_count, frame_size)

    async with httpx.AsyncClient(timeout=120.0) as client:
        sheet_bytes = await _generate_one_image(
            sheet_prompt, client, reference_image_path=reference_image
        )

    sheet_path = work_dir / "sheet.png"
    sheet_path.write_bytes(sheet_bytes)

    # Remove background from the full sheet in one API call, then slice.
    # This gives far cleaner results than per-frame removal — remove.bg sees
    # the whole composition and won't clip edges or eat light-coloured pixels.
    sheet_bg_path = work_dir / "sheet_bg_removed.png"
    await asyncio.to_thread(
        sprite_processor.strip_background,
        str(sheet_path),
        str(sheet_bg_path),
        config.get("removebg_api_key", "") or None,
    )

    raw_paths = await asyncio.to_thread(
        _slice_sheet, str(sheet_bg_path), frame_count, str(raw_dir)
    )

    # Batch canvas-normalize: shared scale factor across all frames keeps
    # character size consistent across poses (no size-pop between frames).
    processed_paths = [str(processed_dir / f"frame_{i}.png") for i in range(len(raw_paths))]
    await asyncio.to_thread(
        sprite_processor.canvas_normalize_batch,
        list(raw_paths),
        processed_paths,
        int(eff["canvas_width"]),
        int(eff["canvas_height"]),
        int(eff["char_width"]),
        int(eff["foot_anchor_y"]),
    )

    # ── Validate frames against reference ────────────────────────────────────
    validation = await asyncio.to_thread(
        _validate_frames, processed_paths, reference_image
    )
    if not validation["all_passed"]:
        logger.warning(
            "generate_game_sprite: %d frame(s) failed validation — "
            "consider regenerating. Details:\n%s",
            len(validation["failed_indices"]),
            validation["summary"],
        )

    contact_sheet_path = work_dir / "contact_sheet.png"
    contact_sheet.make_contact_sheet(
        processed_paths,
        str(contact_sheet_path),
        validation=validation["frames"],
    )

    gif_path: str | None = None
    if preview and processed_paths:
        gif_target = work_dir / "preview.gif"
        # Pillow stitch first — renders frames at their native 100x250 canvas size,
        # so the character is clearly visible. Godot preview was producing a
        # 1280x720 viewport with the sprite as a tiny dot, which is useless for
        # frame review.
        gif_path = None
        try:
            gif_path = await asyncio.to_thread(
                _stitch_gif, processed_paths, str(gif_target), fps
            )
        except Exception as exc:
            logger.warning("Pillow GIF stitch failed (%s) — no preview GIF", exc)

    written = False
    game_path = config.get("game_project_path", "")

    # Always copy contact sheet + GIF into the game project's _preview_tmp/<animation>/
    # so Claude can read them for review regardless of write_to_assets setting.
    if game_path:
        preview_dir = Path(game_path) / "_preview_tmp" / animation
        preview_dir.mkdir(parents=True, exist_ok=True)
        # Step 1: raw model output
        shutil.copy2(str(sheet_path), preview_dir / "step1_raw_sheet.png")
        # Step 2: after bg removal on full sheet
        if sheet_bg_path.exists():
            shutil.copy2(str(sheet_bg_path), preview_dir / "step2_bg_removed_sheet.png")
        # Step 3: individual slices (before canvas normalization)
        for i, raw_p in enumerate(raw_paths):
            shutil.copy2(raw_p, preview_dir / f"step3_raw_frame_{i}.png")
        # Step 4: canvas-normalized frames
        for i, src in enumerate(processed_paths):
            shutil.copy2(src, preview_dir / f"step4_frame_{i}.png")
        # Contact sheet + GIF
        shutil.copy2(str(contact_sheet_path), preview_dir / "contact_sheet.png")
        shutil.copy2(str(sheet_path), preview_dir / "sheet.png")
        for i, src in enumerate(processed_paths):
            shutil.copy2(src, preview_dir / f"frame_{i}.png")
        if gif_path:
            shutil.copy2(gif_path, preview_dir / "preview.gif")

    if write_to_assets:
        if not game_path:
            logger.warning("write_to_assets=True but game_project_path is empty")
        else:
            assets_dir = Path(game_path) / "assets" / "characters" / character / animation
            assets_dir.mkdir(parents=True, exist_ok=True)
            for i, src in enumerate(processed_paths):
                shutil.copy2(src, assets_dir / f"frame_{i}.png")
            written = True

    return {
        "frames": processed_paths,
        "sheet": str(sheet_path),
        "contact_sheet": str(contact_sheet_path),
        "gif": gif_path,
        "written_to_assets": written,
        "reference_image": reference_image,
        "effective_config": eff,
        "validation": validation,
        "validation_summary": validation["summary"],
        "validation_passed": validation["all_passed"],
    }


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
