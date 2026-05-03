"""Single-image sprite processing for game-asset-server.

Mirrors the core pipeline from the game repo's scripts/process_sprites.py:
  1. Background removal — remove.bg API if a key is given, else flood-fill.
  2. Tight crop to character bounding box.
  3. Resize so character is `char_width` px wide.
  4. Place on `canvas_width` x `canvas_height` transparent canvas with
     feet anchored at `foot_anchor_y`.
"""

from __future__ import annotations

import io
import logging
from collections import deque
from pathlib import Path

import httpx
from PIL import Image

logger = logging.getLogger(__name__)


_BG_SAT_THRESHOLD = 25
_BG_MIN_THRESHOLD = 80
_ALPHA_TRANSPARENT = 32


def _remove_bg_api(img: Image.Image, api_key: str) -> Image.Image:
    """Call remove.bg and return an RGBA image with transparent background."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    buf.seek(0)

    response = httpx.post(
        "https://api.remove.bg/v1.0/removebg",
        files={"image_file": ("sprite.png", buf, "image/png")},
        data={"size": "auto"},
        headers={"X-Api-Key": api_key},
        timeout=30.0,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"remove.bg API error {response.status_code}: {response.text[:200]}"
        )
    return Image.open(io.BytesIO(response.content)).convert("RGBA")


def _remove_bg_floodfill(img: Image.Image) -> Image.Image:
    """Flood-fill fallback. BFS from corners, then sweep enclosed islands."""
    rgba = img.convert("RGBA")
    pixels = rgba.load()
    w, h = rgba.size

    def is_bg(x: int, y: int) -> bool:
        r, g, b, a = pixels[x, y]
        if a < _ALPHA_TRANSPARENT:
            return True
        sat = max(r, g, b) - min(r, g, b)
        min_ch = min(r, g, b)
        return min_ch > _BG_MIN_THRESHOLD and sat < _BG_SAT_THRESHOLD

    corners = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    visited = [[False] * h for _ in range(w)]
    queue: deque[tuple[int, int]] = deque()
    for cx, cy in corners:
        if not visited[cx][cy] and is_bg(cx, cy):
            visited[cx][cy] = True
            queue.append((cx, cy))
    while queue:
        x, y = queue.popleft()
        pixels[x, y] = (0, 0, 0, 0)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and not visited[nx][ny] and is_bg(nx, ny):
                visited[nx][ny] = True
                queue.append((nx, ny))

    def still_bg(x: int, y: int) -> bool:
        r, g, b, a = pixels[x, y]
        sat = max(r, g, b) - min(r, g, b)
        min_ch = min(r, g, b)
        return a > _ALPHA_TRANSPARENT and min_ch > _BG_MIN_THRESHOLD and sat < _BG_SAT_THRESHOLD

    visited2 = [[False] * h for _ in range(w)]
    for sx in range(w):
        for sy in range(h):
            if visited2[sx][sy] or not still_bg(sx, sy):
                continue
            component: list[tuple[int, int]] = []
            q: deque[tuple[int, int]] = deque([(sx, sy)])
            visited2[sx][sy] = True
            touches_border = False
            while q:
                x, y = q.popleft()
                component.append((x, y))
                if x == 0 or x == w - 1 or y == 0 or y == h - 1:
                    touches_border = True
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and not visited2[nx][ny] and still_bg(nx, ny):
                        visited2[nx][ny] = True
                        q.append((nx, ny))
            if not touches_border:
                for x, y in component:
                    r, g, b, _ = pixels[x, y]
                    pixels[x, y] = (r, g, b, 0)

    return rgba


def _strip_background(img: Image.Image, api_key: str | None) -> Image.Image:
    if api_key:
        try:
            return _remove_bg_api(img, api_key)
        except Exception as exc:
            logger.warning("remove.bg failed (%s) — falling back to flood-fill", exc)
    return _remove_bg_floodfill(img)


def _fit_to_canvas(
    rgba: Image.Image,
    char_width: int,
    canvas_w: int,
    canvas_h: int,
    foot_y: int,
) -> Image.Image:
    alpha = rgba.split()[-1]
    bbox = alpha.point(lambda v: 255 if v > 16 else 0).getbbox()
    if bbox is None:
        raise RuntimeError("No foreground detected after background removal.")

    cropped = rgba.crop(bbox)
    cw, ch = cropped.size
    scale = char_width / cw
    new_w = char_width
    new_h = max(1, round(ch * scale))
    cropped = cropped.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    paste_x = (canvas_w - new_w) // 2
    paste_y = foot_y - new_h
    canvas.paste(cropped, (paste_x, paste_y), cropped)
    return canvas


def process_image(
    input_path: str,
    output_path: str,
    canvas_width: int,
    canvas_height: int,
    char_width: int,
    foot_anchor_y: int,
    removebg_api_key: str,
) -> str:
    """Process one sprite image end-to-end and write the result to output_path."""
    src = Image.open(input_path)
    stripped = _strip_background(src, removebg_api_key or None)
    canvas = _fit_to_canvas(stripped, char_width, canvas_width, canvas_height, foot_anchor_y)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path
