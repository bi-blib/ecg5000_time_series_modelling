from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scipy.interpolate import make_interp_spline

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "presentation_assets" / "ecg_input_diagram.png"
REGULAR = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

NAVY = "#17324D"
TEAL = "#1E8F7A"
MUTED = "#647486"
GRID = "#DDE6EC"
PALE = "#E8F7F4"

# Color palette for 5 heartbeat classes
CLASS_COLORS = {
    1: "#2878B5",  # Blue
    2: "#1E8F7A",  # Teal
    3: "#E98B2A",  # Green
    4: "#C94C4C",  # Indigo
    5: "#E5B34B",  # Coral
}

CLASS_NAMES = {
    1: "Normal",
    2: "R-on-T PVC",
    3: "PVC",
    4: "SP or Ectopic",
    5: "Unclassified",
}


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

    # Extract representative signal (closest to mean) for each available class
    samples = {}
    for cls_id in sorted(CLASS_NAMES.keys()):
        cls_data = raw[raw[:, 0] == cls_id, 1:]
        if len(cls_data) > 0:
            mean_sig = cls_data.mean(axis=0)
            rep = cls_data[np.argmin(np.square(cls_data - mean_sig).sum(axis=1))]
            samples[cls_id] = rep

    image = Image.new("RGBA", (1700, 620), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    draw.text((52, 34), "Model input: multi-class ECG heartbeats", font=font(39, True), fill=NAVY)

    # Signal panel
    panel = (52, 125, 1195, 525)
    draw.rounded_rectangle(panel, radius=24, fill="white", outline=GRID, width=4)
    
    # Plot region contracted on the right (960) to leave space for the legend
    plot = (120, 160, 960, 475)
    left, top, right, bottom = plot

    # Gridlines
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = left + fraction * (right - left)
        draw.line((x, top, x, bottom), fill=GRID, width=2)
    for fraction in (0.0, 0.5, 1.0):
        y = top + fraction * (bottom - top)
        draw.line((left, y, right, y), fill=GRID, width=2)

    # Compute bounds with tight padding to maximize vertical scale (amplitude)
    all_concat = np.concatenate(list(samples.values()))
    low, high = float(all_concat.min()), float(all_concat.max())
    padding = (high - low) * 0.02
    low, high = low - padding, high + padding

    # Plot each heartbeat line
    for cls_id, signal in samples.items():
        # 1. Original x-indices
        x_orig = np.arange(len(signal))
        
        # 2. Dense x-indices (e.g., 500 smooth points instead of 140)
        x_dense = np.linspace(0, len(signal) - 1, 500)
        
        # 3. Create a cubic spline (k=3) for smooth curves
        spline = make_interp_spline(x_orig, signal, k=3)
        signal_smooth = spline(x_dense)

        # 4. Map dense points to pixel coordinates
        points = []
        for index, value in zip(x_dense, signal_smooth):
            x = left + index / (len(signal) - 1) * (right - left)
            y = bottom - (float(value) - low) / (high - low) * (bottom - top)
            points.append((x, y))
            
        draw.line(points, fill=CLASS_COLORS[cls_id], width=4, joint="curve")

    # Legend inside the panel but cleanly right of the grid
    legend_x, legend_y = 980, 160
    for cls_id, name in CLASS_NAMES.items():
        if cls_id in samples:
            draw.line([(legend_x, legend_y + 12.5), (legend_x + 25, legend_y + 12.5)], fill=CLASS_COLORS[cls_id], width=4)
            draw.text((legend_x + 30, legend_y), f"{name}", font=font(25, True), fill=NAVY)
            legend_y += 30

    # Time step labels below plot box
    draw.text((left, 485), "time step 1", font=font(24), fill=MUTED)
    end_label = "time step 140"
    end_box = draw.textbbox((0, 0), end_label, font=font(24))
    draw.text((right - (end_box[2] - end_box[0]), 485), end_label, font=font(24), fill=MUTED)

    # Amplitude label pushed outside the outer white panel margin
    draw_vertical_text(
        image=image,
        position=(75, 270),  # Adjust x, y coordinates as needed
        text="amplitude",
        font_style=font(24, False),
        fill=MUTED
    )

    # smooth_image = image.resize((image.width // 2, image.height // 2), Image.Resampling.LANCZOS)
    image.save(OUTPUT)

def draw_vertical_text(image, position, text, font_style, fill):
    """Renders text vertically (rotated 90 degrees counter-clockwise)."""
    # Create a temporary transparent image to fit the text
    dummy_draw = ImageDraw.Draw(image)
    bbox = dummy_draw.textbbox((0, 0), text, font=font_style)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    
    txt_img = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    txt_draw = ImageDraw.Draw(txt_img)
    txt_draw.text((-bbox[0], -bbox[1]), text, font=font_style, fill=fill)
    
    # Rotate 90 degrees counter-clockwise
    rotated = txt_img.rotate(90, expand=True)
    
    # Paste rotated image using its alpha channel as a mask
    image.paste(rotated, position, rotated)

if __name__ == "__main__":
    main()

