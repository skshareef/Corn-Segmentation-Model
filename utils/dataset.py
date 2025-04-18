import os
import random
import numpy as np
from PIL import Image
from typing import Optional, Callable, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as T
from .transforms import build_img_transform, build_mask_transform, build_train_aug

class SegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, aug=None, img_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.fnames = sorted(os.listdir(image_dir))
        self.aug = aug
        self.img_tf = img_transform
        self.msk_tf = mask_transform
        self.mask_paths = [self._find_mask_path(f) for f in self.fnames]

    def _find_mask_path(self, img_fname):
        stem, _ = os.path.splitext(img_fname)
        for ext in ("png", "jpg", "tif"):
            for suffix in ["_mask", ""]:
                path = os.path.join(self.mask_dir, f"{stem}{suffix}.{ext}")
                if os.path.isfile(path):
                    return path
        raise FileNotFoundError(f"No mask found for {img_fname}")

    def __len__(self): return len(self.fnames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.fnames[idx])
        msk_path = self.mask_paths[idx]
        img = np.array(Image.open(img_path).convert("RGB"))
        msk = np.array(Image.open(msk_path).convert("L"), dtype=np.uint8)

        if self.aug:
            out = self.aug(image=img, mask=msk)
            img, msk = out["image"], out["mask"]

        img = Image.fromarray(img)
        msk = Image.fromarray(msk)
        if self.img_tf: img = self.img_tf(img)
        if self.msk_tf: msk = self.msk_tf(msk)
        return img, msk


def build_loaders(img_dir, msk_dir, batch_size, val_ratio, test_ratio, num_workers, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    dataset = SegDataset(
        img_dir, msk_dir,
        aug=None,
        img_transform=build_img_transform(),
        mask_transform=build_mask_transform()
    )
    n = len(dataset)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(seed))

    train_ds.dataset.aug = build_train_aug()

    def _make_loader(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=shuffle)

    return _make_loader(train_ds, True), _make_loader(val_ds, False), _make_loader(test_ds, False)
