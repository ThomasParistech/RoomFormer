"""Download preprocessed datasets and pretrained checkpoints for RoomFormer.

Usage:
    python fork/scripts/download_data.py
"""

import os
import subprocess
import zipfile

FORK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DOWNLOADS = [
    ("https://polybox.ethz.ch/index.php/s/wKYWFsQOXHnkwcG/download", "input", "stru3d"),
    ("https://polybox.ethz.ch/index.php/s/VfrJdPvTgG0EBJG/download", "input", "scenecad"),
    ("https://polybox.ethz.ch/index.php/s/vlBo66X0NTrcsTC/download", "input", "checkpoints"),
]


def download_and_extract(url: str, dest_dir: str, name: str) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, f"{name}.zip")

    print(f"Downloading {name}...")
    subprocess.run(["curl", "-L", "-o", zip_path, url], check=True)

    print(f"Extracting {name}...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    os.remove(zip_path)

    print(f"{name} ready at {os.path.join(dest_dir, name)}")


if __name__ == "__main__":
    for url, rel_dir, name in DOWNLOADS:
        dest_dir = os.path.join(FORK_ROOT, rel_dir)
        download_and_extract(url, dest_dir, name)
    print("Done.")
