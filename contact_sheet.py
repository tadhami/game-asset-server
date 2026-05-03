"""Contact sheet generator: tile sprite frames horizontally into one PNG."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

GAP_PX = 8


def make_contact_sheet(frame_paths: list[str], output_path: str) -> str:
    """Tile frames horizontally with a small transparent gap between each."""
    if not frame_paths:
        raise ValueError("frame_paths is empty")

    frames = [Image.open(p).convert("RGBA") for p in frame_paths]
    cell_w = max(f.width for f in frames)
    cell_h = max(f.height for f in frames)

    n = len(frames)
    total_w = n * cell_w + (n - 1) * GAP_PX
    sheet = Image.new("RGBA", (total_w, cell_h), (0, 0, 0, 0))

    x = 0
    for f in frames:
        offset_x = x + (cell_w - f.width) // 2
        offset_y = (cell_h - f.height) // 2
        sheet.paste(f, (offset_x, offset_y), f)
        x += cell_w + GAP_PX

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path
