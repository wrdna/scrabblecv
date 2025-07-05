import os
import shutil
import random
from pathlib import Path

seed = 42
split_ratio = 0.8
label_list = Path("../data/corners/diamond_captures/labels.txt")
src_img_dir = Path("../data/corners/diamond_captures/img/")
src_anno_dir = Path("../data/corners/diamond_captures/anno/")
out_dir = Path("../data/corners/diamond_captures_ultra/")

# Prepare output
if out_dir.exists():
    shutil.rmtree(out_dir)
(out_dir / "images/train").mkdir(parents=True)
(out_dir / "images/val").mkdir(parents=True)
(out_dir / "labels/train").mkdir(parents=True)
(out_dir / "labels/val").mkdir(parents=True)

image_files = sorted(list(src_img_dir.glob("*.jpg")) + list(src_img_dir.glob("*.png")))
image_stems = [img.stem for img in image_files]

random.seed(seed)
random.shuffle(image_stems)
split_idx = int(len(image_stems) * split_ratio)
train_stems = image_stems[:split_idx]
val_stems = image_stems[split_idx:]

def link_set(stems, split):
    for stem in stems:
        # image
        for ext in ['.jpg', '.png']:
            img_src = src_img_dir / f"{stem}{ext}"
            if img_src.exists():
                break
        else:
            continue  # skip if no valid image found

        img_dst = out_dir / "images" / split / img_src.name
        os.symlink(os.path.abspath(img_src), img_dst)

        # label
        label_src = src_anno_dir / f"{stem}.txt"
        label_dst = out_dir / "labels" / split / label_src.name
        if label_src.exists():
            os.symlink(os.path.abspath(label_src), label_dst)

link_set(train_stems, "train")
link_set(val_stems, "val")

with open(label_list) as f:
    class_names = [line.strip() for line in f if line.strip()]

yaml_path = Path("data.yaml")
with open(yaml_path, "w") as f:
    f.write(f"path: {out_dir.resolve()}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write(f"names: {class_names}\n")
