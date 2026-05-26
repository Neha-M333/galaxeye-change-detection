import rasterio
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import albumentations as A
from pathlib import Path


EO_MEAN = [0.485, 0.456, 0.406]
EO_STD  = [0.229, 0.224, 0.225]
SAR_MEAN = 2.9777
SAR_STD  = 1.7104


def get_transforms(split, image_size=256):
    if split == "train":
        return A.Compose([
            A.RandomCrop(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ], additional_targets={"sar": "image"})
    else:
        return A.Compose([
            A.CenterCrop(image_size, image_size),
        ], additional_targets={"sar": "image"})


class ChangeDetectionDataset(Dataset):
    def __init__(self, data_path, split=None, image_size=256,
                 sar_mean=SAR_MEAN, sar_std=SAR_STD):
        """
        Args:
            data_path: path to split folder (e.g. /path/to/test)
                       expects subfolders: pre-event, post-event, target
            split: "train", "val", "test" or None
                   if None, infers from data_path
        """
        self.sar_mean  = sar_mean
        self.sar_std   = sar_std
        self.split     = split if split else Path(data_path).name
        self.transform = get_transforms(self.split, image_size)

        data_path = Path(data_path)

        # Handle nested structure (e.g. test/test/pre-event)
        if (data_path / self.split).exists():
            data_path = data_path / self.split

        self.eo_files   = sorted((data_path / "pre-event").glob("*.tif"))
        self.sar_files  = sorted((data_path / "post-event").glob("*.tif"))
        self.mask_files = sorted((data_path / "target").glob("*.tif"))

        assert len(self.eo_files) > 0, f"No EO files found in {data_path / 'pre-event'}"
        assert len(self.eo_files) == len(self.sar_files) == len(self.mask_files), \
            "Mismatch in file counts!"

        print(f"[{self.split}] {len(self.eo_files)} tiles loaded.")

    def __len__(self):
        return len(self.eo_files)

    def preprocess_eo(self, data):
        data = data.astype(np.float32) / 255.0
        data = data.transpose(1, 2, 0)
        for c in range(3):
            data[:, :, c] = (data[:, :, c] - EO_MEAN[c]) / EO_STD[c]
        return data

    def preprocess_sar(self, data):
        data = data.astype(np.float32)
        data = np.log1p(data)
        data = (data - self.sar_mean) / self.sar_std
        data = data.transpose(1, 2, 0)
        return data

    def __getitem__(self, idx):
        with rasterio.open(self.eo_files[idx])   as src: eo   = src.read()
        with rasterio.open(self.sar_files[idx])  as src: sar  = src.read()
        with rasterio.open(self.mask_files[idx]) as src: mask = src.read(1)

        eo   = self.preprocess_eo(eo)
        sar  = self.preprocess_sar(sar)

        # Label remapping: 0,1 → 0 (no change), 2,3 → 1 (change)
        mask = np.where(mask >= 2, 1, 0).astype(np.float32)

        aug  = self.transform(image=eo, sar=sar, mask=mask)
        eo   = aug["image"]
        sar  = aug["sar"]
        mask = aug["mask"]

        eo   = torch.tensor(eo.transpose(2, 0, 1),  dtype=torch.float32)
        sar  = torch.tensor(sar.transpose(2, 0, 1), dtype=torch.float32)
        mask = torch.tensor(mask,                    dtype=torch.float32)

        return eo, sar, mask


def make_sampler(dataset):
    weights = []
    for mask_file in dataset.mask_files:
        with rasterio.open(mask_file) as src:
            mask = src.read(1)
        mask   = np.where(mask >= 2, 1, 0)
        weight = np.sqrt(mask.mean()) + 0.05
        weights.append(weight)
    weights = torch.tensor(weights, dtype=torch.float32)
    return WeightedRandomSampler(
        weights, num_samples=len(weights), replacement=True)
