from pathlib import Path
from random import Random
import os

root = Path("../../training_data/species_specific")
src = root / "raw"

files = sorted(src.glob("*.wav"))
assert len(files) == 1501, len(files)

Random(42).shuffle(files)

test = files[:150]
val = files[150:300]
train = files[300:]

for split, subset in {
    "train": train,
    "val": val,
    "test": test,
}.items():
    out = root / split
    out.mkdir(parents=True, exist_ok=True)

    for f in subset:
        target = out / f.name
        if not target.exists():
            target.symlink_to(f.resolve())

print("train:", len(train))
print("val:", len(val))
print("test:", len(test))
