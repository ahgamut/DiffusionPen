import xml.etree.ElementTree as ET
from xml.sax.saxutils import unescape as _unescape
import re
import struct
from PIL import Image, ImageDraw
from dataclasses import dataclass


def unescape(x):
    return _unescape(x, {"&quot;": '"', "&apos;": "'"})


def depack_string(blob, offset):
    l = struct.unpack("B", blob[offset : offset + 1])[0]
    l, enc = struct.unpack(f"B{l}s", blob[offset : (offset + l + 1)])
    return l, enc.decode("ascii")


def enpack_string(string):
    enc = string.encode("ascii")
    l = len(enc)
    return l, enc


@dataclass(slots=True)
class Word:
    x_start: int
    x_end: int
    y_start: int
    y_end: int
    pl_width: int
    pl_height: int
    pl_ystart: int
    raw: str
    idd: str
    wid: str

    @property
    def nwidth(self):
        return (self.x_end - self.x_start) / self.pl_width

    @property
    def nheight(self):
        return (self.y_end - self.y_start) / self.pl_height

    @property
    def width(self):
        return self.x_end - self.x_start

    @property
    def height(self):
        return self.y_end - self.y_start

    @property
    def parent_doc(self):
        return "-".join(self.idd.split("-")[:2])

    @property
    def parent_line(self):
        return self.idd.split("-")[2]

    @property
    def inline_index(self):
        return int(self.idd.split("-")[3])

    @property
    def writer_id(self):
        return self.wid

    def __str__(self):
        return f'("{self.raw}", ({self.x_start}, {self.y_start}) -- ({self.x_end}, {self.y_end}))'

    def __repr__(self):
        return f'("{self.raw}", ({self.x_start}, {self.y_start}) -- ({self.x_end}, {self.y_end}))'

    def to_bytes(self):
        l1, enc1 = enpack_string(self.raw)
        l2, enc2 = enpack_string(self.idd)
        l3, enc3 = enpack_string(self.wid)
        blob = struct.pack(
            f"iiiiiiiB{l1}sB{l2}sB{l3}s",
            self.x_start,
            self.x_end,
            self.y_start,
            self.y_end,
            self.pl_width,
            self.pl_height,
            self.pl_ystart,
            l1,
            enc1,
            l2,
            enc2,
            l3,
            enc3,
        )
        return blob

    @classmethod
    def from_bytes(cls, blob):
        offset = 0
        things = struct.unpack("iiiiiii", blob[:28])
        offset += 28
        l1, raw = depack_string(blob, offset)
        offset += l1 + 1  # because l1 is a 'B'
        l2, idd = depack_string(blob, offset)
        offset += l2 + 1  # because l2 is a 'B'
        l3, wid = depack_string(blob, offset)
        return Word(*things, raw, idd, wid)

    @classmethod
    def get_startend(cls, elem, wid):
        parts = [x for x in elem]
        if len(parts) == 0:
            err_string = "?? {}, {}, {}".format(elem, elem.attrib, wid)
            raise RuntimeError("empty attribs: " + err_string)
        elif len(parts) == 1:
            x_start = int(parts[0].attrib["x"])
            x_end = x_start + int(parts[0].attrib["width"])
            y_start = int(parts[0].attrib["y"])
            y_end = y_start + int(parts[0].attrib["height"])
        else:
            x_start = min(int(p.attrib["x"]) for p in parts)
            x_end = max(int(p.attrib["x"]) + int(p.attrib["width"]) for p in parts)
            y_start = min(int(p.attrib["y"]) for p in parts)
            y_end = max(int(p.attrib["y"]) + int(p.attrib["height"]) for p in parts)

        return x_start, y_start, x_end, y_end

    @classmethod
    def from_elem(cls, elem, wid, plw, plh, ply):
        x_start, y_start, x_end, y_end = cls.get_startend(elem, wid)
        raw = unescape(elem.attrib["text"])
        idd = elem.attrib["id"]
        return Word(x_start, x_end, y_start, y_end, plw, plh, ply, raw, idd, wid)


class Prompt:
    def __init__(self, fname):
        self.fname = fname
        tree = ET.parse(fname)
        root = tree.getroot()
        self.root = tree.getroot()
        self.idd = root.attrib["id"]
        self.writer_id = root.attrib["writer-id"]  # wid

        parts = list(x for x in root)
        text_prompt = [unescape(line.attrib["text"]) for line in parts[0]]
        text_prompt = "\n".join(text_prompt)
        text_prompt = re.sub("([^a-zA-Z\d\s:])", " \\1 ", text_prompt)
        # print(text_prompt)
        self.img_width = int(root.attrib["width"])
        self.img_height = int(root.attrib["height"])

        words = []
        for l in parts[1]:
            x0 = 1e5
            x1 = -1
            y0 = 1e5
            y1 = -1
            for w in l.findall("word"):
                xs0, ys0, xs1, ys1 = Word.get_startend(w, self.writer_id)
                x0 = min(x0, xs0)
                y0 = min(y0, ys0)
                x1 = max(x1, xs1)
                y1 = max(y1, ys1)

            plw = max(self.img_width, x1 - x0)
            plh = y1 - y0
            for w in l.findall("word"):
                words.append(Word.from_elem(w, self.writer_id, plw, plh, y0))
        # print(words)

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

        for w in self.words:
            w.pl_width = self.width

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
    p = Prompt("saved_iam_data/a06-124.xml")
    for w1 in p.words:
        blob1 = w1.to_bytes()
        w2 = Word.from_bytes(blob1)
        blob2 = w2.to_bytes()
        print(str(w1) == str(w2), blob1 == blob2)
        print(w1, w2)
    # raw = Image.open("try-gen/a06-124.png").convert("RGB")
    # crop = p.get_anno_crop(raw)
    # crop.save("pls.png")


if __name__ == "__main__":
    main()
