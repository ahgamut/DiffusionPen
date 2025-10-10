import json
from PIL import Image

PUNCT_ST = set(",./;:'\"[]!@#$%^&*()-_+=\\|")


def get_resize_placed_word(
    word, img, placings, units_per_em, font_size=16, dpi=300, use_aspect=False
):
    width = 0
    height = -1
    ystart = -1
    yend = 1e6

    for ch in word:
        plc = placings[ch]
        width += plc["ink_width"]  # advance_width?
        # height = max(height, plc["ink_height"])
        ystart = max(ystart, plc["yMax"])
        yend = min(yend, plc["yMin"])
        height = max(height, ystart - yend)

    # placings assume origin is bottom-left
    # but, PIL assume origin is top-left
    ystart = units_per_em - ystart

    z = (font_size / units_per_em) * (dpi / 72)

    pixwidth = int(width * z)
    pixheight = int(height * z)
    pixystart = int(ystart * z)

    if use_aspect:
        # do we need pixwidth?
        aspect_ratio = img.width / img.height
        pixwidth = int(pixheight * aspect_ratio)

    # print(word, pixwidth, pixheight)
    pixwidth = max(pixwidth, 3)
    pixheight = max(pixheight, 3)
    img_rsz = img.resize((pixwidth, pixheight))

    wcs = set([ch for ch in word])
    if len(wcs - PUNCT_ST) == 0:
        punct = True
    else:
        punct = False

    return dict(ystart=pixystart, rsz=img_rsz, punct=punct)


def build_placed_paragraph(
    words, fakes, max_line_width=900, font_size=16, dpi=300, use_aspect=True
):
    assert len(words) == len(fakes)
    N = len(words)

    with open("utils/char_placing.json", "r") as fp:
        cpj = json.load(fp)

    units_per_em = cpj["upm"]
    placings = cpj["glyphs"]
    conversion_factor = (font_size / units_per_em) * (dpi / 72)

    width_of_space = int(conversion_factor * placings[" "]["advance_width"])
    height_of_line = int(conversion_factor * units_per_em)

    cur_width = 0
    cur_line = []

    lines = []

    for i in range(N):
        word = words[i]
        img = fakes[i]  # this has been thru crop_whitespace_width
        wdata = get_resize_placed_word(
            word, img, placings, units_per_em, font_size, dpi, use_aspect
        )

        if wdata["punct"]:
            if cur_width + wdata["rsz"].width > max_line_width:
                lines.append(cur_line)
                cur_line = []
                cur_width = 0
        else:
            if cur_width + width_of_space + wdata["rsz"].width > max_line_width:
                lines.append(cur_line)
                cur_line = []
                cur_width = 0

        if not wdata["punct"]:
            cur_width += width_of_space

        cur_line.append(wdata)
        cur_width += wdata["rsz"].width

    para_height = (2 + len(lines)) * height_of_line
    para_width = max_line_width
    paragraph_image = Image.new(
        mode="RGB", size=(para_width, para_height), color="white"
    )

    for i, line in enumerate(lines):
        pl_ystart = i * height_of_line
        cur_width = 0
        for wdata in line:
            if not wdata["punct"]:
                cur_width += width_of_space
            paragraph_image.paste(
                wdata["rsz"], (cur_width, pl_ystart + wdata["ystart"])
            )
            cur_width += wdata["rsz"].width

    paragraph_image = paragraph_image.convert("L")
    return paragraph_image
