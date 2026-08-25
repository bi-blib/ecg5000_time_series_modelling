"""Render a compact input diagram using a real ECG5000 heartbeat."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "presentation_assets" / "ecg_input_diagram.png"
REGULAR = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

NAVY = "#17324D"
TEAL = "#1E8F7A"
BLUE = "#2878B5"
MUTED = "#647486"
GRID = "#DDE6EC"
PALE = "#E8F7F4"


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else REGULAR, size)


def centered(draw, box, text, text_font, fill):
    left, top, right, bottom = box
    bounds = draw.multiline_textbbox((0, 0), text, font=text_font, spacing=6, align="center")
    width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    draw.multiline_text(
        ((left + right - width) / 2, (top + bottom - height) / 2 - bounds[1]),
        text,
        font=text_font,
        fill=fill,
        spacing=6,
        align="center",
    )


def main():
    raw = np.loadtxt(ROOT / "ECG5000" / "ECG5000_TRAIN.txt")
    # Use a representative normal heartbeat nearest to the class mean.
    normal = raw[raw[:, 0] == 1, 1:]
    mean_signal = normal.mean(axis=0)
    sample = normal[np.argmin(np.square(normal - mean_signal).sum(axis=1))]

    image = Image.new("RGBA", (1700, 620), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    draw.text((52, 34), "Model input: one complete ECG heartbeat", font=font(39, True), fill=NAVY)

    # Signal panel.
    panel = (52, 125, 1195, 525)
    draw.rounded_rectangle(panel, radius=24, fill="white", outline=GRID, width=4)
    plot = (120, 185, 1130, 455)
    left, top, right, bottom = plot

    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = left + fraction * (right - left)
        draw.line((x, top, x, bottom), fill=GRID, width=2)
    for fraction in (0.0, 0.5, 1.0):
        y = top + fraction * (bottom - top)
        draw.line((left, y, right, y), fill=GRID, width=2)

    low, high = float(sample.min()), float(sample.max())
    padding = (high - low) * 0.12
    low, high = low - padding, high + padding
    points = []
    for index, value in enumerate(sample):
        x = left + index / (len(sample) - 1) * (right - left)
        y = bottom - (float(value) - low) / (high - low) * (bottom - top)
        points.append((x, y))
    draw.line(points, fill=BLUE, width=6, joint="curve")

    draw.text((left, 470), "time step 1", font=font(20), fill=MUTED)
    end_label = "time step 140"
    end_box = draw.textbbox((0, 0), end_label, font=font(20))
    draw.text((right - (end_box[2] - end_box[0]), 470), end_label, font=font(20), fill=MUTED)
    draw.text((68, 278), "amplitude", font=font(19, True), fill=MUTED)

    # Arrow and model-ready tensor.
    draw.line((1218, 325, 1320, 325), fill=TEAL, width=7)
    draw.polygon([(1320, 325), (1289, 306), (1289, 344)], fill=TEAL)

    tensor = (1350, 183, 1648, 470)
    draw.rounded_rectangle(tensor, radius=24, fill=PALE, outline=TEAL, width=5)
    centered(draw, (1370, 205, 1628, 355), "140 time steps\n×\n1 feature", font(31, True), NAVY)
    centered(draw, (1370, 365, 1628, 442), "tensor [140, 1]", font(24, True), TEAL)

    draw.text(
        (52, 560),
        "The label is separate; the encoder receives only the ordered amplitude sequence.",
        font=font(24),
        fill=MUTED,
    )
    image.save(OUTPUT)


if __name__ == "__main__":
    main()
