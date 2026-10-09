"""Download the three external corpora used to test recognition beyond UCI HAR, and unpack them where the scripts read them.

    HAR70+  18 adults aged 70-95, lower-back accelerometer (Ustad et al., 2023; UCI dataset 780, CC BY 4.0)
            -> external/har70/har70plus/*.csv
    HARTH   22 adults aged 25-68, same sensor and protocol (Logacjov et al., 2021; UCI dataset 779, CC BY 4.0)
            -> external/harth/harth/*.csv
    KU-HAR  trimmed recordings of 90 adults, smartphone in a waist bag (Sikder & Nahid, 2021; Mendeley Data
            10.17632/45f952y38r.5, file "2.Trimmed_interpolated_data.zip", CC BY 4.0)
            -> external/kuhar/<activity>/*.csv

A corpus already present is skipped. About 520 MB are downloaded in total.
"""
import os, urllib.request, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = os.path.join(ROOT, "external")
SOURCES = [
    ("har70", "https://archive.ics.uci.edu/static/public/780/har70.zip", "har70.zip",
     os.path.join("har70", "har70plus")),
    ("harth", "https://archive.ics.uci.edu/static/public/779/harth.zip", "harth.zip",
     os.path.join("harth", "harth")),
    ("kuhar", "https://data.mendeley.com/public-files/datasets/45f952y38r/files/"
              "49c6120b-59fd-466c-97da-35d53a4be595/file_downloaded", "kuhar_trimmed.zip",
     os.path.join("kuhar", "0.Stand")),
]


def main():
    os.makedirs(EXT, exist_ok=True)
    for name, url, zname, probe in SOURCES:
        if os.path.isdir(os.path.join(EXT, probe)):
            print("%s: present, skipped" % name, flush=True)
            continue
        zpath = os.path.join(EXT, zname)
        if not os.path.exists(zpath):
            print("%s: downloading %s" % (name, url), flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "trimedai-reproduce"})
            with urllib.request.urlopen(req, timeout=600) as r, open(zpath + ".part", "wb") as f:
                while True:
                    block = r.read(1 << 20)
                    if not block:
                        break
                    f.write(block)
            os.replace(zpath + ".part", zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(os.path.join(EXT, name))
        assert os.path.isdir(os.path.join(EXT, probe)), "unexpected layout in %s" % zname
        print("%s: unpacked to external/%s" % (name, name), flush=True)


if __name__ == "__main__":
    main()
