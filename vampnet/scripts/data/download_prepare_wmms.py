#!/usr/bin/env python3
"""
Download and prepare the Watkins Marine Mammal Sound Database (WMMS)
"Best Of" collection for WhAM/VampNet domain adaptation.

Default output layout (run from wham/vampnet):
    training_data/
    ├── downloads/wmms_bestof/              # downloaded ZIP archives
    ├── domain_adaptation/train/wmms/        # standardized WAV files
    ├── domain_adaptation/val/wmms/          # standardized WAV files
    └── manifests/wmms_bestof_manifest.csv

The script:
  1. Queries the Internet Archive mirror metadata to discover all ZIP files.
  2. Downloads the ZIP files with retries and size checks.
  3. Extracts audio, including audio inside nested ZIP files.
  4. Creates a deterministic, species-stratified train/validation split.
  5. Converts audio to mono, 44.1-kHz, 16-bit PCM WAV with ffmpeg.
  6. Writes a provenance and validation manifest.

Requirements:
    pip install requests
    ffmpeg and ffprobe available on PATH
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import random
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


IA_IDENTIFIER = "watkins_best_of_whales_202008"
IA_METADATA_URL = f"https://archive.org/metadata/{IA_IDENTIFIER}"
IA_DOWNLOAD_BASE = f"https://archive.org/download/{IA_IDENTIFIER}"

AUDIO_EXTENSIONS = {".wav", ".aif", ".aiff", ".flac", ".mp3", ".ogg", ".m4a", ".mp4"}


@dataclass(frozen=True)
class ArchiveFile:
    name: str
    size_bytes: int | None


@dataclass(frozen=True)
class ExtractedAudio:
    archive_name: str
    member_name: str
    local_path: Path


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"^best[_\-\s]*of[_\-\s]*", "", value)
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "unknown_species"


def species_from_archive(archive_name: str) -> str:
    return slugify(Path(archive_name).stem)


def stable_seed(base_seed: int, text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return base_seed ^ int(digest[:16], 16)


def build_http_session() -> requests.Session:
    retry = Retry(
        total=6,
        connect=6,
        read=6,
        status=6,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
    session = requests.Session()
    session.mount("https://", adapter)
    session.headers.update(
        {"User-Agent": "WhAM-WMMS-data-preparation/1.0 (research use)"}
    )
    return session


def check_command(command: str) -> None:
    if shutil.which(command) is None:
        raise RuntimeError(
            f"Required command '{command}' was not found on PATH. "
            "Install ffmpeg before continuing."
        )


def discover_archives(session: requests.Session) -> list[ArchiveFile]:
    response = session.get(IA_METADATA_URL, timeout=120)
    response.raise_for_status()
    payload = response.json()

    archives: list[ArchiveFile] = []
    for item in payload.get("files", []):
        name = str(item.get("name", ""))
        if not name.lower().endswith(".zip"):
            continue

        raw_size = item.get("size")
        try:
            size_bytes = int(raw_size) if raw_size is not None else None
        except (TypeError, ValueError):
            size_bytes = None

        archives.append(ArchiveFile(name=name, size_bytes=size_bytes))

    archives.sort(key=lambda x: x.name.lower())
    if not archives:
        raise RuntimeError(
            "The Internet Archive metadata response contained no ZIP files. "
            "The item may be unavailable or its layout may have changed."
        )
    return archives


def human_size(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "unknown"
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TiB"


def download_archive(
    session: requests.Session,
    archive: ArchiveFile,
    destination_dir: Path,
) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / Path(archive.name).name

    if destination.exists():
        current_size = destination.stat().st_size
        if archive.size_bytes is None or current_size == archive.size_bytes:
            print(f"[download] exists, skipping: {destination.name}")
            return destination
        print(
            f"[download] size mismatch for {destination.name}: "
            f"{current_size} != {archive.size_bytes}; downloading again"
        )

    encoded_name = quote(archive.name, safe="/")
    url = f"{IA_DOWNLOAD_BASE}/{encoded_name}"
    partial = destination.with_suffix(destination.suffix + ".part")

    print(f"[download] {archive.name} ({human_size(archive.size_bytes)})")
    with session.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        with partial.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)

    if archive.size_bytes is not None and partial.stat().st_size != archive.size_bytes:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Incomplete download for {archive.name}: "
            f"expected {archive.size_bytes} bytes."
        )

    partial.replace(destination)
    return destination


def unique_temp_path(
    destination_dir: Path,
    archive_name: str,
    member_name: str,
) -> Path:
    suffix = Path(member_name).suffix.lower()
    stem = slugify(Path(member_name).stem)
    digest = hashlib.sha256(
        f"{archive_name}::{member_name}".encode("utf-8")
    ).hexdigest()[:12]
    return destination_dir / f"{stem}__{digest}{suffix}"


def extract_zip_stream(
    zip_source: Path | io.BytesIO,
    archive_name: str,
    destination_dir: Path,
    prefix: str = "",
) -> list[ExtractedAudio]:
    extracted: list[ExtractedAudio] = []

    with zipfile.ZipFile(zip_source) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue

            member = info.filename
            normalized = member.replace("\\", "/")
            if normalized.startswith("__MACOSX/"):
                continue

            logical_name = f"{prefix}{normalized}"
            suffix = Path(normalized).suffix.lower()

            if suffix in AUDIO_EXTENSIONS:
                target = unique_temp_path(destination_dir, archive_name, logical_name)
                with zf.open(info) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                extracted.append(
                    ExtractedAudio(
                        archive_name=archive_name,
                        member_name=logical_name,
                        local_path=target,
                    )
                )

            elif suffix == ".zip":
                nested_bytes = io.BytesIO(zf.read(info))
                try:
                    extracted.extend(
                        extract_zip_stream(
                            nested_bytes,
                            archive_name=archive_name,
                            destination_dir=destination_dir,
                            prefix=f"{logical_name}!/",
                        )
                    )
                except zipfile.BadZipFile:
                    print(f"[warning] invalid nested ZIP skipped: {logical_name}")

    return extracted


def deterministic_split(
    records: list[ExtractedAudio],
    species: str,
    val_fraction: float,
    seed: int,
) -> dict[str, str]:
    ordered = sorted(records, key=lambda x: x.member_name.lower())
    n_files = len(ordered)

    # Keep very small species entirely in training. For species with at least
    # five clips, reserve at least one clip for validation.
    if n_files < 5 or val_fraction <= 0:
        n_val = 0
    else:
        n_val = max(1, int(round(n_files * val_fraction)))
        n_val = min(n_val, n_files - 1)

    rng = random.Random(stable_seed(seed, species))
    val_indices = set(rng.sample(range(n_files), n_val))

    return {
        record.member_name: ("val" if idx in val_indices else "train")
        for idx, record in enumerate(ordered)
    }


def output_name(record: ExtractedAudio) -> str:
    stem = slugify(Path(record.member_name).stem)
    digest = hashlib.sha256(
        f"{record.archive_name}::{record.member_name}".encode("utf-8")
    ).hexdigest()[:12]
    return f"{stem}__{digest}.wav"


def run_checked(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def convert_to_wav(source: Path, destination: Path, sample_rate: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 44:
        return

    temporary = destination.with_suffix(".tmp.wav")
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-map_metadata",
        "-1",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(temporary),
    ]

    try:
        run_checked(command)
    except subprocess.CalledProcessError as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"ffmpeg failed for {source.name}:\n{exc.stderr.strip()}"
        ) from exc

    temporary.replace(destination)


def probe_audio(path: Path) -> tuple[float, int, int]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels:format=duration",
        "-of",
        "default=noprint_wrappers=1",
        str(path),
    ]
    result = run_checked(command)

    values: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()

    duration = float(values["duration"])
    sample_rate = int(values["sample_rate"])
    channels = int(values["channels"])
    return duration, sample_rate, channels


def prepare_archive(
    archive_path: Path,
    train_root: Path,
    val_root: Path,
    val_fraction: float,
    seed: int,
    sample_rate: int,
) -> list[dict[str, str]]:
    species = species_from_archive(archive_path.name)
    print(f"[prepare] {archive_path.name} -> species '{species}'")

    manifest_rows: list[dict[str, str]] = []

    with tempfile.TemporaryDirectory(prefix="wmms_extract_") as tmp:
        temp_dir = Path(tmp)
        try:
            extracted = extract_zip_stream(
                archive_path,
                archive_name=archive_path.name,
                destination_dir=temp_dir,
            )
        except zipfile.BadZipFile as exc:
            raise RuntimeError(f"Invalid ZIP archive: {archive_path}") from exc

        if not extracted:
            print(f"[warning] no supported audio found in {archive_path.name}")
            return manifest_rows

        split_map = deterministic_split(
            extracted,
            species=species,
            val_fraction=val_fraction,
            seed=seed,
        )

        for index, record in enumerate(extracted, start=1):
            split = split_map[record.member_name]
            root = val_root if split == "val" else train_root
            destination = root / species / output_name(record)

            try:
                convert_to_wav(record.local_path, destination, sample_rate)
                duration, actual_sr, channels = probe_audio(destination)
            except Exception as exc:
                print(
                    f"[warning] skipped {record.member_name} "
                    f"because validation failed: {exc}"
                )
                continue

            if duration <= 0:
                destination.unlink(missing_ok=True)
                print(f"[warning] zero-duration file removed: {record.member_name}")
                continue

            manifest_rows.append(
                {
                    "dataset": "wmms_best_of",
                    "mirror_item": IA_IDENTIFIER,
                    "source_archive": archive_path.name,
                    "source_member": record.member_name,
                    "species": species,
                    "split": split,
                    "output_path": str(destination),
                    "duration_seconds": f"{duration:.6f}",
                    "sample_rate": str(actual_sr),
                    "channels": str(channels),
                }
            )

            if index % 50 == 0 or index == len(extracted):
                print(f"  processed {index}/{len(extracted)}")

    return manifest_rows


def write_manifest(rows: list[dict[str, str]], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "mirror_item",
        "source_archive",
        "source_member",
        "species",
        "split",
        "output_path",
        "duration_seconds",
        "sample_rate",
        "channels",
    ]

    rows = sorted(
        rows,
        key=lambda row: (
            row["split"],
            row["species"],
            row["source_archive"],
            row["source_member"],
        ),
    )

    with manifest_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: Iterable[dict[str, str]]) -> None:
    summary = {
        "train": {"files": 0, "seconds": 0.0},
        "val": {"files": 0, "seconds": 0.0},
    }
    species = set()

    for row in rows:
        split = row["split"]
        summary[split]["files"] += 1
        summary[split]["seconds"] += float(row["duration_seconds"])
        species.add(row["species"])

    print("\nWMMS preparation completed")
    print("-" * 48)
    for split in ("train", "val"):
        files = int(summary[split]["files"])
        hours = summary[split]["seconds"] / 3600
        print(f"{split.capitalize():<12} files: {files:>6}")
        print(f"{split.capitalize():<12} hours: {hours:>9.3f}")
    print(f"{'Species':<12}: {len(species):>6}")
    total_files = sum(int(summary[s]["files"]) for s in summary)
    total_hours = sum(summary[s]["seconds"] for s in summary) / 3600
    print(f"{'Total files':<12}: {total_files:>6}")
    print(f"{'Total hours':<12}: {total_hours:>9.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and prepare the WMMS Best Of collection for WhAM."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="WhAM VampNet working directory containing training_data (default: .)",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.10,
        help="Validation fraction within each species archive (default: 0.10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Deterministic split seed (default: 2026)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        help="Output WAV sample rate (default: 44100)",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="List discovered ZIP archives without downloading them",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N archives for a smoke test",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing train/wmms and val/wmms directories first",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not 0 <= args.val_fraction < 1:
        raise ValueError("--val-fraction must satisfy 0 <= value < 1")
    if args.sample_rate <= 0:
        raise ValueError("--sample-rate must be positive")

    root = args.root.expanduser().resolve()
    training_data = root / "training_data"
    archive_dir = training_data / "downloads" / "wmms_bestof"
    train_root = training_data / "domain_adaptation" / "train" / "wmms"
    val_root = training_data / "domain_adaptation" / "val" / "wmms"
    manifest_path = training_data / "manifests" / "wmms_bestof_manifest.csv"

    session = build_http_session()
    archives = discover_archives(session)

    if args.limit is not None:
        archives = archives[: args.limit]

    total_download = sum(x.size_bytes or 0 for x in archives)
    print(f"Discovered {len(archives)} ZIP archives")
    print(f"Known total download size: {human_size(total_download)}")
    for archive in archives:
        print(f"  {archive.name:<50} {human_size(archive.size_bytes)}")

    if args.list_only:
        return 0

    check_command("ffmpeg")
    check_command("ffprobe")

    if args.clean:
        shutil.rmtree(train_root, ignore_errors=True)
        shutil.rmtree(val_root, ignore_errors=True)

    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, str]] = []

    for number, archive in enumerate(archives, start=1):
        print(f"\n=== Archive {number}/{len(archives)} ===")
        archive_path = download_archive(session, archive, archive_dir)
        rows = prepare_archive(
            archive_path=archive_path,
            train_root=train_root,
            val_root=val_root,
            val_fraction=args.val_fraction,
            seed=args.seed,
            sample_rate=args.sample_rate,
        )
        all_rows.extend(rows)

    write_manifest(all_rows, manifest_path)
    print_summary(all_rows)
    print(f"Manifest: {manifest_path}")
    print(f"Train:    {train_root}")
    print(f"Val:      {val_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)