#!/usr/bin/env python3
"""Generate semantic class/color legends used by eval semantic plots."""

import argparse
import json
import math
from pathlib import Path
from typing import List

from imgviz import label_colormap
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    "MP3D": {
        "class_info": REPO_ROOT / "configs/MP3D/class_info_file.json",
        "num_classes": 41,
        "columns": 3,
    },
    "Replica": {
        "class_info": REPO_ROOT / "configs/Replica/office0/class_info_file.json",
        "num_classes": 102,
        "columns": 4,
    },
}


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu") / filename,
        Path("/usr/share/fonts/dejavu") / filename,
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def load_class_names(path: Path, num_classes: int) -> List[str]:
    with path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    names = ["unknown/background"] * num_classes
    for class_id, value in metadata.items():
        index = int(class_id)
        if 0 <= index < num_classes:
            names[index] = value["name"]
    return names


def draw_legend(dataset: str, output_dir: Path) -> Path:
    config = DATASETS[dataset]
    num_classes = config["num_classes"]
    columns = config["columns"]
    names = load_class_names(config["class_info"], num_classes)
    colors = label_colormap(num_classes)[:, :3]

    width = 1920
    margin_x = 60
    gap_x = 28
    title_height = 112
    header_height = 42
    row_height = 60
    footer_height = 70
    rows = math.ceil(num_classes / columns)
    content_width = width - 2 * margin_x - (columns - 1) * gap_x
    column_width = content_width // columns
    height = title_height + header_height + rows * row_height + footer_height

    image = Image.new("RGB", (width, height), (247, 248, 250))
    draw = ImageDraw.Draw(image)
    title_font = load_font(34, bold=True)
    header_font = load_font(18, bold=True)
    text_font = load_font(19)
    id_font = load_font(18, bold=True)
    footer_font = load_font(16)

    title = f"{dataset} Semantic Class Color Legend"
    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(
        ((width - (title_box[2] - title_box[0])) // 2, 34),
        title,
        fill=(24, 28, 35),
        font=title_font,
    )

    for column in range(columns):
        x = margin_x + column * (column_width + gap_x)
        draw.text((x, title_height), "COLOR", fill=(80, 86, 96), font=header_font)
        draw.text((x + 62, title_height), "ID", fill=(80, 86, 96), font=header_font)
        draw.text((x + 112, title_height), "CLASS LABEL", fill=(80, 86, 96), font=header_font)
        draw.text(
            (x + column_width - 90, title_height),
            "HEX",
            fill=(80, 86, 96),
            font=header_font,
        )

    for class_id, (name, color) in enumerate(zip(names, colors)):
        column = class_id // rows
        row = class_id % rows
        x = margin_x + column * (column_width + gap_x)
        y = title_height + header_height + row * row_height
        rgb = tuple(int(channel) for channel in color)
        hex_color = "#{:02X}{:02X}{:02X}".format(*rgb)

        if row % 2 == 0:
            draw.rounded_rectangle(
                (x - 8, y - 4, x + column_width, y + row_height - 6),
                radius=5,
                fill=(238, 240, 243),
            )
        draw.rectangle((x, y + 4, x + 44, y + 48), fill=rgb, outline=(42, 45, 52), width=2)
        draw.text((x + 62, y + 14), f"{class_id:>3}", fill=(32, 35, 42), font=id_font)
        draw.text((x + 112, y + 14), name, fill=(32, 35, 42), font=text_font)
        hex_box = draw.textbbox((0, 0), hex_color, font=text_font)
        draw.text(
            (x + column_width - (hex_box[2] - hex_box[0]), y + 14),
            hex_color,
            fill=(32, 35, 42),
            font=text_font,
        )

    footer = f"Source: imgviz.label_colormap({num_classes}) | Values shown as RGB"
    footer_box = draw.textbbox((0, 0), footer, font=footer_font)
    draw.text(
        ((width - (footer_box[2] - footer_box[0])) // 2, height - 44),
        footer,
        fill=(88, 94, 104),
        font=footer_font,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset.lower()}_semantic_color_legend.png"
    image.save(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "semantic_legends",
    )
    args = parser.parse_args()

    for dataset in DATASETS:
        output_path = draw_legend(dataset, args.output_dir)
        print(output_path)


if __name__ == "__main__":
    main()
