import json
from PIL import Image

def get_resize_placed_word(word, img, placings, units_per_em, font_size=16, dpi=300):
    width = 0
    height = -1
    ystart = -1

    for ch in word:
        plc = placings[ch]
        width += plc["ink_width"]  # advance_width?
        height = max(height, plc["ink_height"])
        ystart = max(ystart, plc["yMax"])

    # placings assume origin is bottom-left
    # but, PIL assume origin is top-left
    ystart = units_per_em - ystart

    z = (font_size / units_per_em) * (dpi / 72)

    pixwidth = int(width * z)
    pixheight = int(height * z)
    pixystart = int(ystart * z)

    # do we need pixwidth?
    aspect_ratio = img.width / img.height
    pixwidth = int(pixheight * aspect_ratio)

    img_rsz = img.resize((pixwidth, pixheight))

    return dict(ystart=ystart, rsz=img_rsz)


def build_placed_paragraph(fakes, words, max_line_width=900, font_size=16, dpi=300):
    assert len(words) == len(fakes)
    N = len(words)

    with open("utils/char_placing.json", "r") as fp:
        cpj = json.load(fp)

    units_per_em = cpj["upm"]
    placings = cpj["glyphs"]
    conversion_factor = (font_size / units_per_em) * (dpi / 72)

    width_of_space = int(conversion_factor * placings[" "]["width"])
    height_of_line = int(conversion_factor * upm)

    cur_width = 0
    cur_line = []

    lines = []

    cur_width += width_of_space
    for i in range(N):
        word = words[i]
        img = fakes[i]  # this has been thru crop_whitespace_width
        wdata = get_resize_placed_word(word, img, placings, units_per_em, font_size, dpi)
        if cur_width + wdata["rsz"].width > max_line_width:
            lines.append(cur_line)
            cur_line = []
            cur_width = 0

        cur_line.append(wdata)
        cur_width += wdata["rsz"].width
        cur_width += width_of_space

    para_height = len(lines) * height_of_line
    para_width = max_line_width
    paragraph_image = Image.new(mode="RGB", size=(para_width, para_height), color="white")

    for i, line in enumerate(lines):
        pl_ystart = i * height_of_line
        cur_width = width_of_space
        for wdata in line:
            paragraph_image.paste(wdata["rsz"], (cur_width, pl_ystart + wdata["ystart"]))
            cur_width += wdata["rsz"].width
            cur_width += width_of_space

    paragraph_image = paragraph_image.convert("L")
    return paragraph_image
