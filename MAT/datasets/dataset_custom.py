import os
import cv2
import random
import numpy as np
import torch
from PIL import Image

class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, path, use_labels=True, max_size=None, xflip=False, resolution=512):
        self.path = path
        self.images_path = os.path.join(path, "images")
        self.masks_path = os.path.join(path, "masks")
        self.labels_path = os.path.join(path, "labels")
        self.use_lables = use_labels
        self.max_size = max_size
        self.xflip = xflip
        self.res = resolution
        self.num_dataset = len(os.listdir(self.images_path))
        self._get_label_index()
        self._get_sample_item()

    def _get_sample_item(self):
        self.sample_img, self.sample_mask, self.sample_label = self.__getitem__(0)

    def _get_label_index(self):
        with open(os.path.join(self.path, "labels.txt"), "r") as f:
            labels = f.read().split(", ")
        self.label_idx = {label: i for i, label in enumerate(labels)}

    @property
    def resolution(self):
        return self.res
    
    @property
    def has_labels(self):
        return self.use_lables
    
    @property
    def label_shape(self):
        return list(self.sample_label.shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]
    
    @property
    def name(self):
        return "places265"
    
    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]
    
    @property
    def image_shape(self):
        return list(self.sample_img.shape)
    
    @property
    def has_onehot_labels(self):
        return True

    def __len__(self):
        return self.num_dataset

    def __getitem__(self, idx):
        img_name = str(idx) + ".jpg"
        label_name = str(idx) + ".txt"
        img_path = os.path.join(self.images_path, img_name)
        mask_path = os.path.join(self.masks_path, img_name)
        label_path = os.path.join(self.labels_path, label_name)

        img = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path))

        # restricted to 512x512
        res = 512
        H, W, C = img.shape
        # Padding
        if H < res or W < res:
            top = 0
            bottom = max(0, res - H)
            left = 0
            right = max(0, res - W)
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT)
        
        H, W, _ = img.shape
        h = random.randint(0, H - res)
        w = random.randint(0, W - res)
        img = img[h:h+res, w:w+res, :]
        mask = mask[h:h+res, w:w+res]

        img = np.ascontiguousarray(img.transpose(2, 0, 1)) # HWC => CHW

        with open(label_path, "r") as f:
            class_name = f.read().strip()
        label = np.zeros(len(self.label_idx.keys()), dtype=np.int8)
        label[self.label_idx[class_name]] = 1

        return img.copy(), mask.copy(), label
    
if __name__ == "__main__":
    # root_dir = os.path.join("FirstBatch", "combined", "train_dir")
    # dataset = CustomDataset(path=root_dir, resolution=512)
    # print(dataset[0])
    # print(dataset[6000])
    pass