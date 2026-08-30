from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "nepali_data"
SOURCES = ["munamadan.txt", "ramayan.txt", "Sumnima.txt"]
OUT = DATA_DIR / "merged.txt"


def main():
    parts = []
    for name in SOURCES:
        path = DATA_DIR / name
        text = path.read_text(encoding="utf-8")
        parts.append(text)
        print(f"{name}: {len(text)} chars")
    merged = "\n".join(parts)
    OUT.write_text(merged, encoding="utf-8")
    print(f"wrote {OUT} ({len(merged)} chars)")


if __name__ == "__main__":
    main()
