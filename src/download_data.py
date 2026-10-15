"""Download the raw election datasets into data/raw (skips files already present)."""
import sys
import urllib.request

from config import DATA_SOURCES, RAW_DIR


def download(force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in DATA_SOURCES.items():
        dest = RAW_DIR / name
        if dest.exists() and not force:
            print(f"[skip] {name} already exists")
            continue
        print(f"[get ] {name}")
        urllib.request.urlretrieve(url, dest)


if __name__ == "__main__":
    download(force="--force" in sys.argv)
