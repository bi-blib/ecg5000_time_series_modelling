"""Create presentation-ready PNG diagrams for the ECG5000 task slide."""

from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path(__file__).resolve().parent
FONT_REGULAR = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

NAVY = "#17324D"
BLUE = "#2878B5"
CYAN = "#52B6D8"
TEAL = "#1E8F7A"
ORANGE = "#E98B2A"
RED = "#C94C4C"
GREY = "#596775"
GRID = "#E4EAEF"


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size)


def centered_multiline(draw, box, text, text_font, fill, spacing=8):
    left, top, right, bottom = box
    bbox = draw.multiline_textbbox((0, 0), text, font=text_font, spacing=spacing, align="center")
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    draw.multiline_text(
        ((left + right - width) / 2, (top + bottom - height) / 2 - bbox[1]),
        text,
        font=text_font,
        fill=fill,
        spacing=spacing,
        align="center",
    )


def rounded_box(draw, box, title, detail, face, edge):
    draw.rounded_rectangle(box, radius=28, fill=face, outline=edge, width=5)
    left, top, right, bottom = box
    height = bottom - top
    centered_multiline(draw, (left + 20, top + 26, right - 20, top + height * 0.53), title, font(36, True), NAVY)
    centered_multiline(draw, (left + 20, top + height * 0.48, right - 20, bottom - 20), detail, font(25), GREY, 7)


def arrow(draw, start, end, color=BLUE, width=7):
    x1, y1 = start
    x2, y2 = end
    draw.line((x1, y1, x2, y2), fill=color, width=width)
    angle = math.atan2(y2 - y1, x2 - x1)
    length, wing = 24, 0.55
    points = [
        (x2, y2),
        (x2 - length * math.cos(angle - wing), y2 - length * math.sin(angle - wing)),
        (x2 - length * math.cos(angle + wing), y2 - length * math.sin(angle + wing)),
    ]
    draw.polygon(points, fill=color)


def make_pipeline():
    image = Image.new("RGBA", (2880, 864), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    draw.text((58, 38), "From heartbeat signal to screening decision", font=font(49, True), fill=NAVY)

    rounded_box(draw, (58, 278, 570, 680), "ECG5000", "5,000 heartbeats\n140 time steps each", "#EAF4FA", BLUE)
    rounded_box(draw, (785, 278, 1330, 680), "Preprocessing", "Train-only scaling\nreshape to [N, 140, 1]", "#E8F7F4", TEAL)
    rounded_box(draw, (1545, 278, 2060, 680), "Sequence encoder", "Temporal representation\nof one heartbeat", "#F2EEFA", "#7654A6")
    rounded_box(draw, (2290, 140, 2820, 468), "Stage 1", "Normal vs abnormal", "#FFF2E5", ORANGE)
    rounded_box(draw, (2290, 535, 2820, 835), "Stage 2", "Abnormal class\n2 / 3 / 4 / 5", "#FCECEC", RED)

    arrow(draw, (570, 479), (785, 479))
    arrow(draw, (1330, 479), (1545, 479))
    arrow(draw, (2060, 460), (2290, 310))
    arrow(draw, (2060, 500), (2290, 670), RED)
    draw.text((2072, 626), "if abnormal", font=font(24, True), fill=RED)
    image.save(OUT_DIR / "ecg_pipeline.png")


def make_distribution():
    image = Image.new("RGB", (1890, 1044), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 42), "ECG5000 class distribution", font=font(56, True), fill=NAVY)
    draw.text((70, 116), "The rarest class represents only 0.5% of the dataset", font=font(32), fill=GREY)
    draw.text(
        (70, 168),
        "Implemented split:\n500 train  |  2,250 validation  |  2,250 test\n(stratified validation/test)",
        font=font(30, True),
        fill=BLUE,
        spacing=7,
    )

    labels = [
        "Normal beat",
        "R-on-T premature ventricular contraction",
        "Premature ventricular contraction",
        "Supraventricular premature or ectopic beat",
        "Unclassified beat",
    ]
    counts = [2919, 1767, 96, 194, 24]
    percentages = [58.4, 35.3, 1.9, 3.9, 0.5]
    colors = [BLUE, "#1E8F7A", ORANGE, RED, "#E5B34B"]

    # Donut chart. Draw smallest slices last so their boundaries remain clear.
    pie_box = (60, 300, 790, 1030)
    start_angle = -90
    for count, color in zip(counts, colors):
        sweep = count / sum(counts) * 360
        draw.pieslice(pie_box, start=start_angle, end=start_angle + sweep, fill=color, outline="white")
        start_angle += sweep

    # Cut out the center to produce a spacious, slide-friendly donut.
    center_box = (260, 500, 590, 830)
    draw.ellipse(center_box, fill="white")
    centered_multiline(draw, center_box, "5,000\nheartbeats", font(44, True), NAVY, 5)

    # Large diagnosis card aligned with the title and containing exact values.
    diagnosis_box = (915, 35, 1845, 815)
    draw.rounded_rectangle(diagnosis_box, radius=28, fill="#F7FAFC", outline=GRID, width=4)
    legend_x, legend_y, row_gap = 955, 125, 133
    draw.text((legend_x, 55), "Diagnosis", font=font(42, True), fill=GREY)
    for index, (label, count, pct, color) in enumerate(zip(labels, counts, percentages, colors)):
        y = legend_y + index * row_gap
        draw.rounded_rectangle((legend_x, y, legend_x + 64, y + 64), radius=14, fill=color)
        draw.text((legend_x + 88, y - 3), label, font=font(29, True), fill=NAVY)
        draw.text((legend_x + 88, y + 47), f"{pct:.1f}%  |  {count:,} beats", font=font(29), fill=GREY)

    # Make the central comparison explicit without putting unreadable labels on tiny slices.
    comparison_box = (915, 830, 1845, 925)
    draw.rounded_rectangle(comparison_box, radius=22, fill="#F5F8FA", outline=GRID, width=4)
    centered_multiline(
        draw,
        comparison_box,
        "Three rare diagnoses together: only 6.3% of the data",
        font(29, True),
        NAVY,
    )

    note = "Implication: oversampling + macro-F1 evaluation"
    note_font = font(29, True)
    note_box = draw.textbbox((0, 0), note, font=note_font)
    note_width = note_box[2] - note_box[0]
    implication_box = (1845 - note_width - 64, 940, 1845, 1030)
    draw.rounded_rectangle(implication_box, radius=22, fill="#FFF5F5", outline="#F1CACA", width=4)
    centered_multiline(draw, implication_box, note, note_font, RED)
    image.save(OUT_DIR / "ecg_class_distribution.png")


if __name__ == "__main__":
    make_pipeline()
    make_distribution()
