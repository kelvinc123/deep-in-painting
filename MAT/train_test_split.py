import os
import shutil
import random
import fnmatch
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

DATAPATH = "./data/FirstBatch/"
TRAIN_PROP = 0.9

images_path = os.path.join(DATAPATH, "images")
labels_path = os.path.join(DATAPATH, "labels")
output_path = os.path.join(DATAPATH, "combined")
os.makedirs(os.path.join(output_path, "train_dir", "images"), exist_ok=True)
os.makedirs(os.path.join(output_path, "train_dir", "masks"), exist_ok=True)
os.makedirs(os.path.join(output_path, "train_dir", "labels"), exist_ok=True)
os.makedirs(os.path.join(output_path, "val_dir", "images"), exist_ok=True)
os.makedirs(os.path.join(output_path, "val_dir", "masks"), exist_ok=True)
os.makedirs(os.path.join(output_path, "val_dir", "labels"), exist_ok=True)
counter = 0
train_results = []
val_results = []
label_sets = set()
for img in tqdm(os.listdir(images_path)):
    class_name = img.split("_")[0]
    label_sets.add(class_name)
    img_path = os.path.join(images_path, img)
    fn_pattern = f"{os.path.splitext(img)[0]}*.jpg"
    labels = [label for label in os.listdir(labels_path) if fnmatch.fnmatch(label, fn_pattern)]
    for label in labels:
        label_path = os.path.join(labels_path, label)

        # restricted to 512x512
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(label_path))
        res = 512
        H, W, C = img.shape
        # Padding
        if H < res or W < res:
            top = 0
            bottom = max(0, res - H)
            left = 0
            right = max(0, res - W)
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT)
            mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_REFLECT)
        
        H, W, _ = img.shape
        h = random.randint(0, H - res)
        w = random.randint(0, W - res)
        img = img[h:h+res, w:w+res, :]
        mask = mask[h:h+res, w:w+res]

        u = random.uniform(0, 1)
        if u < TRAIN_PROP:
            Image.fromarray(img, "RGB").save(os.path.join(output_path, "train_dir", "images", f"{counter}.jpg"))
            Image.fromarray(mask).save(os.path.join(output_path, "train_dir", "masks", f"{counter}.jpg"))
            with open(os.path.join(output_path, "train_dir", "labels", f"{counter}.txt"), "w") as f:
                f.write(class_name)
            train_results.append(
                {
                    "idx": counter,
                    "class": class_name,
                    "orignal_img_path": img_path,
                    "original_mask_path": label_path
                }
            )
        else:
            Image.fromarray(img, "RGB").save(os.path.join(output_path, "val_dir", "images", f"{counter}.jpg"))
            Image.fromarray(mask).save(os.path.join(output_path, "val_dir", "masks", f"{counter}.jpg"))
            with open(os.path.join(output_path, "val_dir", "labels", f"{counter}.txt"), "w") as f:
                f.write(class_name)

            val_results.append(
                {
                    "idx": counter,
                    "class": class_name,
                    "orignal_img_path": img_path,
                    "original_mask_path": label_path
                }
            )
        counter += 1

pd.DataFrame(train_results).to_csv(os.path.join(output_path, "train_summary.csv"), index=False)
pd.DataFrame(val_results).to_csv(os.path.join(output_path, "val_summary.csv"), index=False)
with open(os.path.join(output_path, "train_dir", "labels.txt"), "w") as f:
    f.write(
        ", ".join(list(label_sets))
    )
with open(os.path.join(output_path, "val_dir", "labels.txt"), "w") as f:
    f.write(
        ", ".join(list(label_sets))
    )