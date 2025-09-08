import xml.etree.ElementTree as ET
from xml.sax.saxutils import unescape as _unescape
import re
from PIL import Image, ImageDraw


def unescape(x):
    return _unescape(x, {"&quot;": '"', "&apos;": "'"})


class Word:
    def __init__(self, elem):
        parts = [x for x in elem]
        if len(parts) == 1:
            self.x_start = int(parts[0].attrib["x"])
            self.x_end = self.x_start + int(parts[0].attrib["width"])
            self.y_start = int(parts[0].attrib["y"])
            self.y_end = self.y_start + int(parts[0].attrib["height"])
        else:
            self.x_start = min(int(p.attrib["x"]) for p in parts)
            self.x_end = max(int(p.attrib["x"]) + int(p.attrib["width"]) for p in parts)
            self.y_start = min(int(p.attrib["y"]) for p in parts)
            self.y_end = max(
                int(p.attrib["y"]) + int(p.attrib["height"]) for p in parts
            )

        self.raw = unescape(elem.attrib["text"])
        self.id = elem.attrib["id"]
        self.width = self.x_end - self.x_start
        self.height = self.y_end - self.y_start

    def __str__(self):
        return f'("{self.raw}", ({self.x_start}, {self.y_start}) -- ({self.x_end}, {self.y_end}))'

    def __repr__(self):
        return f'("{self.raw}", ({self.x_start}, {self.y_start}) -- ({self.x_end}, {self.y_end}))'


class Prompt:
    def __init__(self, fname):
        self.fname = fname
        tree = ET.parse(fname)
        root = tree.getroot()
        self.root = tree.getroot()

        parts = list(x for x in root)
        text_prompt = [unescape(line.attrib["text"]) for line in parts[0]]
        text_prompt = "\n".join(text_prompt)
        text_prompt = re.sub("([^a-zA-Z\d\s:])", " \\1 ", text_prompt)
        # print(text_prompt)

        words = []
        for l in parts[1]:
            for w in l:
                if w.tag == "word":
                    words.append(Word(w))
                else:
                    # this is a line contour
                    pass
        # print(words)

        self.id = root.attrib["id"]
        self.writer_id = root.attrib["writer-id"]  # wid
        self.img_width = int(root.attrib["width"])
        self.img_height = int(root.attrib["height"])
        self.text_prompt = text_prompt
        self.words = words

        self.x_start = min(w.x_start for w in self.words)
        self.x_start = max(0, self.x_start - 10)

        self.x_end = max(w.x_end for w in self.words)
        self.x_end = min(self.img_width, self.x_end + 10)

        self.y_start = min(w.y_start for w in self.words)
        self.y_start = max(0, self.y_start - 10)

        self.y_end = max(w.y_end for w in self.words)
        self.y_end = min(self.img_height, self.y_end + 10)

        self.width = self.x_end - self.x_start
        self.height = self.y_end - self.y_start

    def get_cropped(self, img):
        cropped = img.crop((self.x_start, self.y_start, self.x_end, self.y_end))
        return cropped

    def get_anno_crop(self, img):
        dupe = img.convert("RGB")

        cd = ImageDraw.Draw(dupe)
        for w in self.words:
            cd.rectangle(
                [(w.x_start, w.y_start), (w.x_end, w.y_end)], width=2, outline="red"
            )

        cropped = dupe.crop((self.x_start, self.y_start, self.x_end, self.y_end))
        return cropped


def main():
    p = Prompt("try-gen/a06-124.xml")
    raw = Image.open("try-gen/a06-124.png").convert("RGB")
    crop = p.get_anno_crop(raw)
    crop.save("pls.png")


if __name__ == "__main__":
    main()
