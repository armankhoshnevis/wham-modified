from __future__ import annotations

import csv
import os
from pathlib import Path


DOWNLOAD_ROOT = Path("training_data/downloads/fsd50k")
AUDIO_ROOT = DOWNLOAD_ROOT / "FSD50K.dev_audio"
LABELS_CSV = DOWNLOAD_ROOT / "FSD50K.ground_truth" / "dev.csv"

OUTPUT_ROOT = Path("training_data/domain_adaptation")
MANIFEST_ROOT = Path("training_data/manifests")

# This operational definition reproduces the previously identified
# 3,159-recording FSD50K animal subset.
ANIMAL_BRANCHES = {
    "Domestic_animals_and_pets",
    "Livestock_and_farm_animals_and_working_animals",
    "Wild_animals",
}

EXPECTED_COUNTS = {
    "train": 2862,
    "val": 297,
}


def main() -> None:
    if not AUDIO_ROOT.is_dir():
        raise FileNotFoundError(
            f"Audio directory does not exist: {AUDIO_ROOT}"
        )

    if not LABELS_CSV.is_file():
        raise FileNotFoundError(
            f"Ground-truth CSV does not exist: {LABELS_CSV}"
        )

    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)

    # Key by fname so a recording cannot be counted twice.
    selected: dict[str, dict[str, dict[str, str]]] = {
        "train": {},
        "val": {},
    }

    with LABELS_CSV.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        required = {"fname", "labels", "split"}
        missing = required - set(reader.fieldnames or [])

        if missing:
            raise RuntimeError(
                f"Missing CSV columns: {sorted(missing)}"
            )

        for row in reader:
            fname = row["fname"].strip()
            split = row["split"].strip()

            labels = {
                label.strip()
                for label in row["labels"].split(",")
                if label.strip()
            }

            if labels.isdisjoint(ANIMAL_BRANCHES):
                continue

            if split not in selected:
                raise RuntimeError(
                    f"Unexpected split value: {split!r}"
                )

            selected[split][fname] = row

    counts = {
        split: len(records)
        for split, records in selected.items()
    }

    print(f"Selected unique rows: {counts}")

    if counts != EXPECTED_COUNTS:
        raise RuntimeError(
            "The selected FSD50K counts are unexpected.\n"
            f"Expected: {EXPECTED_COUNTS}\n"
            f"Found:    {counts}\n"
            "No output links have been created."
        )

    manifest_fields = [
        "path",
        "source_path",
        "fname",
        "split",
        "labels",
    ]

    created = {
        "train": 0,
        "val": 0,
    }

    existing = {
        "train": 0,
        "val": 0,
    }

    for split, records in selected.items():
        destination_dir = OUTPUT_ROOT / split / "fsd50k"
        destination_dir.mkdir(parents=True, exist_ok=True)

        expected_names = {
            f"{fname}.wav"
            for fname in records
        }

        # Do not silently delete anything unexpected.
        unexpected = [
            path
            for path in destination_dir.glob("*.wav")
            if path.name not in expected_names
        ]

        if unexpected:
            preview = "\n".join(
                str(path)
                for path in unexpected[:10]
            )
            raise RuntimeError(
                f"Unexpected WAV entries exist in {destination_dir}.\n"
                f"Examples:\n{preview}\n"
                "Nothing was deleted. Inspect the directory first."
            )

        manifest_path = (
            MANIFEST_ROOT / f"fsd50k_animal_{split}.csv"
        )

        with manifest_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as manifest_file:
            writer = csv.DictWriter(
                manifest_file,
                fieldnames=manifest_fields,
            )
            writer.writeheader()

            for fname in sorted(records, key=int):
                row = records[fname]

                source = AUDIO_ROOT / f"{fname}.wav"
                if not source.is_file():
                    raise FileNotFoundError(
                        f"Selected source WAV is missing: {source}"
                    )

                destination = destination_dir / source.name
                source_resolved = source.resolve(strict=True)

                if destination.is_symlink():
                    try:
                        destination_target = (
                            destination.resolve(strict=True)
                        )
                    except FileNotFoundError:
                        raise RuntimeError(
                            f"Broken symlink already exists: "
                            f"{destination}"
                        )

                    if destination_target != source_resolved:
                        raise RuntimeError(
                            f"Existing symlink points to the wrong file:\n"
                            f"{destination}\n"
                            f"Current target: {destination_target}\n"
                            f"Expected:       {source_resolved}"
                        )

                    existing[split] += 1

                elif destination.exists():
                    raise RuntimeError(
                        f"A regular file already exists at:\n"
                        f"{destination}\n"
                        "Nothing was overwritten."
                    )

                else:
                    relative_target = os.path.relpath(
                        source_resolved,
                        start=destination.parent.resolve(),
                    )

                    destination.symlink_to(relative_target)
                    created[split] += 1

                writer.writerow(
                    {
                        "path": destination.as_posix(),
                        "source_path": source.as_posix(),
                        "fname": fname,
                        "split": split,
                        "labels": row["labels"],
                    }
                )

    print()
    print("FSD50K animal subset prepared successfully")
    print("-" * 48)
    print(f"Train files:          {counts['train']}")
    print(f"Validation files:     {counts['val']}")
    print(f"Total files:          {sum(counts.values())}")
    print(f"Created symlinks:     {created}")
    print(f"Existing symlinks:    {existing}")
    print(f"Manifest directory:   {MANIFEST_ROOT}")


if __name__ == "__main__":
    main()
