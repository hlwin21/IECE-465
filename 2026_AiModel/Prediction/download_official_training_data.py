import os
from pathlib import Path
import requests

DATA_DIR = Path.home() / "Downloads" / "465data" / "training_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "cbecs2018_public.csv": "https://www.eia.gov/consumption/commercial/data/2018/xls/cbecs2018_final_public.csv",
    "recs2020_public_v7.csv": "https://www.eia.gov/consumption/residential/data/2020/csv/recs2020_public_v7.csv",
}

def download_file(url: str, output_path: Path) -> None:
    print(f"Downloading {output_path.name} ...")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    print(f"Saved to {output_path}")

def main():
    for filename, url in FILES.items():
        out = DATA_DIR / filename
        if out.exists():
            print(f"Already exists: {out}")
        else:
            download_file(url, out)

    print("\nDone.")
    print("Files downloaded to:")
    print(DATA_DIR)

if __name__ == "__main__":
    main()
