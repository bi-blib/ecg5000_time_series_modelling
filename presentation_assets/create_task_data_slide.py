"""Render the ECG5000 task-and-data presentation slide as a 16:9 PNG."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).resolve().parent / "ecg_task_data_slide.png"
REGULAR = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

NAVY = "#17324D"
BLUE = "#2878B5"
TEAL = "#1E8F7A"
PURPLE = "#7654A6"
ORANGE = "#E98B2A"
RED = "#C94C4C"
TEXT = "#33475B"
MUTED = "#647486"
LINE = "#DDE6EC"
BG = "#F7F9FB"


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else REGULAR, size)


def wrapped_lines(draw, text, text_font, max_width):
    words = text.split()
    lines, current = [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        width = draw.textbbox((0, 0), candidate, font=text_font)[2]
        if width <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_wrapped(draw, xy, text, text_font, fill, max_width, line_gap=10):
    x, y = xy
    line_height = draw.textbbox((0, 0), "Ag", font=text_font)[3]
    for line in wrapped_lines(draw, text, text_font, max_width):
        draw.text((x, y), line, font=text_font, fill=fill)
        y += line_height + line_gap
    return y


def card(draw, box, number, title, accent, body, footnote=None):
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=26, fill="white", outline=LINE, width=3)
    draw.rounded_rectangle((left, top, left + 13, bottom), radius=7, fill=accent)

    draw.ellipse((left + 34, top + 31, left + 84, top + 81), fill=accent)
    number_font = font(25, True)
    number_box = draw.textbbox((0, 0), str(number), font=number_font)
    draw.text(
        (left + 59 - (number_box[2] - number_box[0]) / 2, top + 56 - (number_box[3] - number_box[1]) / 2 - number_box[1]),
        str(number),
        font=number_font,
        fill="white",
    )
    draw.text((left + 103, top + 34), title.upper(), font=font(22, True), fill=accent)
    body_end = draw_wrapped(draw, (left + 38, top + 108), body, font(23), TEXT, right - left - 76, 8)
    if footnote:
        draw_wrapped(draw, (left + 38, max(body_end + 12, bottom - 58)), footnote, font(18, True), accent, right - left - 76, 4)


def main():
    image = Image.new("RGB", (1920, 1080), BG)
    draw = ImageDraw.Draw(image)

    draw.text((82, 58), "ECG5000 heartbeat classification", font=font(51, True), fill=NAVY)
    draw.text(
        (84, 127),
        "Task, target, inputs, relevance, data, split, preprocessing and success metric",
        font=font(27),
        fill=MUTED,
    )
    draw.rounded_rectangle((1570, 66, 1838, 126), radius=30, fill="#EAF4FA")
    pill = "SUPERVISED LEARNING"
    pill_box = draw.textbbox((0, 0), pill, font=font(20, True))
    draw.text((1704 - (pill_box[2] - pill_box[0]) / 2, 85), pill, font=font(20, True), fill=BLUE)

    card(
        draw,
        (82, 208, 498, 560),
        1,
        "Task",
        BLUE,
        "Classify one complete heartbeat using a temporal sequence encoder.",
        "Many-to-one classification",
    )
    card(
        draw,
        (528, 208, 944, 560),
        2,
        "Target",
        PURPLE,
        "Stage 1: normal or abnormal. Stage 2: one of four abnormal diagnoses.",
    )
    card(
        draw,
        (974, 208, 1390, 560),
        3,
        "Inputs",
        TEAL,
        "One aligned, univariate ECG beat: 140 ordered amplitude measurements, represented as [140, 1].",
    )
    card(
        draw,
        (1420, 208, 1838, 560),
        4,
        "Practical relevance",
        ORANGE,
        "Automated screening can flag abnormal beats and prioritize recordings for expert review.",
        "Decision support — not diagnosis",
    )
    card(
        draw,
        (82, 600, 498, 952),
        5,
        "Data characteristics",
        RED,
        "5,000 beats and five diagnoses; no missing values. Strong imbalance: class shares range from 58.4% to 0.5%.",
    )
    card(
        draw,
        (528, 600, 944, 952),
        6,
        "Split",
        BLUE,
        "Current code: 500 train, 2,250 validation and 2,250 test beats. The validation/test division is stratified.",
        "10% / 45% / 45%",
    )
    card(
        draw,
        (974, 600, 1390, 952),
        7,
        "Preprocessing",
        TEAL,
        "Re-index labels; fit z-score scaling on training data only; reshape to [N, 140, 1]; apply class-weighted loss.",
        "Prevents leakage and limits majority bias",
    )
    card(
        draw,
        (1420, 600, 1838, 952),
        8,
        "Success metric",
        PURPLE,
        # Macro-F1 computes F1 independently for every diagnosis and then
        # averages those scores equally. This prevents the large normal and
        # R-on-T classes from dominating the headline result, while requiring
        # both useful precision and useful recall for the rare diagnoses.
        "Primary: macro-F1, balancing precision and recall while weighting every diagnosis equally. Also report macro-AUPRC and confusion matrices.",
        "Imbalance-aware evaluation",
    )

    draw.line((82, 1000, 1838, 1000), fill=LINE, width=3)
    draw.text(
        (82, 1019),
        "Core challenge:",
        font=font(23, True),
        fill=RED,
    )
    draw.text(
        (260, 1019),
        "severe class imbalance → class-weighted loss + macro-F1 evaluation",
        font=font(23),
        fill=TEXT,
    )

    image.save(OUT)


if __name__ == "__main__":
    main()
