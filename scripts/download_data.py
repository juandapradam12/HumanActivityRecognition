#!/usr/bin/env python3
"""Download the official UCI HAR partitions (features, subjects, optional raw IMU)."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

UCI_URLS = [
    "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip",
]

FILES = {
    "X_train.txt": "UCI HAR Dataset/train/X_train.txt",
    "y_train.txt": "UCI HAR Dataset/train/y_train.txt",
    "subject_train.txt": "UCI HAR Dataset/train/subject_train.txt",
    "X_test.txt": "UCI HAR Dataset/test/X_test.txt",
    "y_test.txt": "UCI HAR Dataset/test/y_test.txt",
    "subject_test.txt": "UCI HAR Dataset/test/subject_test.txt",
    "features.txt": "UCI HAR Dataset/features.txt",
    "activity_labels.txt": "UCI HAR Dataset/activity_labels.txt",
}

RAW_CHANNELS = (
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Directory for tabular UCI files (default: repo root)",
    )
    parser.add_argument(
        "--with-raw",
        action="store_true",
        help="Also extract raw Inertial Signals into data/inertial_signals/",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite existing files",
    )
    return parser.parse_args()


def download_zip(tmp: Path) -> Path:
    zip_path = tmp / "uci_har.zip"
    last_error = None
    for url in UCI_URLS:
        try:
            print(f"Downloading {url}")
            urlretrieve(url, zip_path)
            return zip_path
        except Exception as exc:  # noqa: BLE001 - try next mirror
            last_error = exc
            print(f"  failed: {exc}")
    raise RuntimeError(f"Could not download UCI HAR dataset: {last_error}")


def extract_inner_dataset(zip_path: Path, tmp: Path) -> zipfile.ZipFile:
    """UCI now ships a nested zip; support both nested and flat layouts."""
    outer = zipfile.ZipFile(zip_path)
    names = outer.namelist()
    nested = [n for n in names if n.endswith("UCI HAR Dataset.zip")]
    if nested:
        inner_path = tmp / "UCI HAR Dataset.zip"
        with outer.open(nested[0]) as src, open(inner_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        return zipfile.ZipFile(inner_path)
    return outer


def _extract_member(archive: zipfile.ZipFile, member: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(member) as src, open(target, "wb") as dst:
        shutil.copyfileobj(src, dst)


def raw_targets(repo_root: Path) -> list[tuple[str, Path]]:
    """Return (archive_member, dest_path) pairs for raw IMU files."""
    out = repo_root / "data" / "inertial_signals"
    pairs = []
    for split in ("train", "test"):
        for channel in RAW_CHANNELS:
            member = (
                f"UCI HAR Dataset/{split}/Inertial Signals/{channel}_{split}.txt"
            )
            pairs.append((member, out / f"{channel}_{split}.txt"))
    return pairs


def main() -> int:
    args = parse_args()
    dest: Path = args.dest
    dest.mkdir(parents=True, exist_ok=True)

    missing_tabular = [
        name for name in FILES if args.force or not (dest / name).exists()
    ]
    missing_raw: list[tuple[str, Path]] = []
    if args.with_raw:
        missing_raw = [
            (member, path)
            for member, path in raw_targets(dest)
            if args.force or not path.exists()
        ]

    if not missing_tabular and not missing_raw:
        print(f"All requested data files already present under {dest}")
        return 0

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        zip_path = download_zip(tmp)
        archive = extract_inner_dataset(zip_path, tmp)
        name_index = {
            Path(n).name: n
            for n in archive.namelist()
            if not Path(n).name.startswith("._")
        }

        for filename in missing_tabular:
            member = FILES[filename]
            if member not in archive.namelist():
                member = name_index.get(filename)
            if member is None:
                print(f"WARNING: {filename} not found in archive", file=sys.stderr)
                continue
            print(f"Extracting {filename}")
            _extract_member(archive, member, dest / filename)

        for member, path in missing_raw:
            if member not in archive.namelist():
                print(f"WARNING: {member} not found", file=sys.stderr)
                continue
            print(f"Extracting raw {path.name}")
            _extract_member(archive, member, path)

    print(f"Done. Tabular data → {dest}")
    if args.with_raw:
        print(f"Raw IMU signals → {dest / 'data' / 'inertial_signals'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
