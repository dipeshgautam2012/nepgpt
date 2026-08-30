import urllib.request
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[1] / "nepali_data"

# /stream/ is the HTML viewer. /download/ is the raw .txt (what curl gets).
FILES = {
    "https://archive.org/download/muna-madan-book-laxmi-prasad-devkota/MunaMadanBookLaxmiPrasadDevkota_djvu.txt": "munamadan.txt",
    "https://archive.org/download/BPKoirala2027BSSumnima/BPKoirala2027BS_Sumnima_djvu.txt": "Sumnima.txt",
    "https://archive.org/download/nepali-bhanubhakta-ramayan/NepaliBhanubhaktaRamayan_djvu.txt": "ramayan.txt",
}


def download(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
    with urllib.request.urlopen(req) as resp, dest.open("wb") as f:
        f.write(resp.read())


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for url, name in FILES.items():
        dest = OUT_DIR / name
        print(f"downloading {name}")
        download(url, dest)
        print(f"  wrote {dest} ({dest.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
