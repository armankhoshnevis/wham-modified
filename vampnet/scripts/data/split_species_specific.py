from pathlib import Path
from random import Random
import os


# This script is stored under:
# vampnet/scripts/data/split_species_specific.py
vampnet_root = Path(__file__).resolve().parents[2]
root = vampnet_root / "training_data" / "species_specific"
src = root / "raw"

files = sorted(src.glob("*.wav"))

if len(files) != 1501:
    raise RuntimeError(
        f"Expected 1501 WAV files in {src}, but found {len(files)}."
    )

Random(42).shuffle(files)

test = files[:150]
val = files[150:300]
train = files[300:]

splits = {
    "train": train,
    "val": val,
    "test": test,
}

for split, subset in splits.items():
    out = root / split
    out.mkdir(parents=True, exist_ok=True)

    expected_names = {f.name for f in subset}

    # Reject unexpected regular WAV files rather than overwriting them.
    for existing in out.glob("*.wav"):
        if existing.name not in expected_names:
            if existing.is_symlink():
                existing.unlink()
            else:
                raise RuntimeError(
                    f"Unexpected regular file in split directory: {existing}"
                )

    for source in subset:
        target = out / source.name
        expected_source = source.resolve(strict=True)

        # is_symlink() remains True even for a broken symlink.
        if target.is_symlink():
            try:
                current_source = target.resolve(strict=True)
            except FileNotFoundError:
                print(f"Replacing broken symlink: {target}")
                target.unlink()
            else:
                if current_source == expected_source:
                    continue

                print(f"Replacing incorrect symlink: {target}")
                target.unlink()

        elif target.exists():
            raise RuntimeError(
                f"Regular file already exists and will not be overwritten: "
                f"{target}"
            )

        relative_source = os.path.relpath(
            expected_source,
            start=out.resolve(),
        )

        target.symlink_to(relative_source)

print("train:", len(train))
print("val:", len(val))
print("test:", len(test))