
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import rasterio
import albumentations as A
import yaml

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def remap_labels(mask):
    """Remap 0,1→0 (no-change)  2,3→1 (change)"""
    binary = np.zeros_like(mask, dtype=np.float32)
    binary[mask == 2] = 1.0
    binary[mask == 3] = 1.0
    return binary

def read_tif(path, is_grayscale=False):
    with rasterio.open(path) as src:
        if is_grayscale:
            img = src.read(1).astype(np.float32)
        else:
            img = src.read().astype(np.float32)
            img = np.transpose(img, (1, 2, 0))   # HWC
    return img

def normalize_eo(img):
    """ImageNet normalization for RGB EO imagery."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img  = img / 255.0 if img.max() > 1.0 else img
    return (img - mean) / (std + 1e-6)

def normalize_sar(img):
    """
    FIX: robust log-scale normalization for SAR backscatter.
    
    Why log? SAR amplitude has Rayleigh/Weibull distribution with 
    heavy tail — linear percentile clipping loses structural detail.
    Log-transform compresses dynamic range, making statistics compatible
    with the EO encoder's expected input distribution (~N(0,1)).
    
    Steps:
      1. Clip to valid positive range (avoid log(0))
      2. Log-transform
      3. Percentile clip in log-domain to remove speckle outliers
      4. Z-score normalize → mean≈0, std≈1  (matches ImageNet EO stats)
    """
    img = np.clip(img, 1e-6, None)          # avoid log(0)
    img = np.log(img + 1e-6)                # log-transform
    p2, p98 = np.percentile(img, 2), np.percentile(img, 98)
    img = np.clip(img, p2, p98)             # clip outliers in log-domain
    mu  = (p2 + p98) / 2.0
    sig = (p98 - p2) / 4.0 + 1e-6          # ≈ 2σ covers [p2,p98]
    return (img - mu) / sig                  # zero-mean, unit-ish scale

class EOSARDataset(Dataset):
    def __init__(self, data_dir, image_size=512, augment=False):
        self.pre_dir    = os.path.join(data_dir, "pre-event")
        self.post_dir   = os.path.join(data_dir, "post-event")
        self.target_dir = os.path.join(data_dir, "target")
        self.filenames  = sorted(os.listdir(self.pre_dir))
        self.image_size = image_size
        self.augment    = augment

        # Pre-compute change ratios for sampler weighting
        print(f"Computing change ratios for {len(self.filenames)} samples...")
        self.change_ratios = []
        for fname in self.filenames:
            mask = read_tif(os.path.join(self.target_dir, fname),
                            is_grayscale=True)
            self.change_ratios.append(float(remap_labels(mask).mean()))

        if augment:
            self.transform = A.Compose([
                A.RandomCrop(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Transpose(p=0.3),
                # Mild geometric distortions
                A.OneOf([
                    A.ElasticTransform(alpha=80, sigma=8, p=1.0),
                    A.GridDistortion(num_steps=5, distort_limit=0.2, p=1.0),
                    A.OpticalDistortion(distort_limit=0.3, p=1.0),
                ], p=0.3),
                # Colour/intensity jitter for EO channel only (applied later)
                # SAR-specific: speckle noise simulation
                A.GaussNoise(var_limit=(0.001, 0.01), p=0.3),
            ])
        else:
            self.transform = A.Compose([
                A.CenterCrop(image_size, image_size),
            ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname    = self.filenames[idx]
        pre_eo   = read_tif(os.path.join(self.pre_dir,    fname), is_grayscale=False)
        post_sar = read_tif(os.path.join(self.post_dir,   fname), is_grayscale=True)
        mask     = read_tif(os.path.join(self.target_dir, fname), is_grayscale=True)

        pre_eo   = normalize_eo(pre_eo)            # HWC, float32, ~N(0,1)
        post_sar = normalize_sar(post_sar)         # HW,  float32, ~N(0,1)  ← FIX
        mask     = remap_labels(mask)              # HW, {0,1}

        post_3d  = post_sar[:, :, np.newaxis]      # HW1
        combined = np.concatenate([pre_eo, post_3d], axis=2)  # HW4
        aug      = self.transform(image=combined, mask=mask)
        combined = aug["image"]
        mask     = aug["mask"]

        eo   = torch.from_numpy(combined[:, :, :3]).permute(2, 0, 1).float()
        sar  = torch.from_numpy(combined[:, :, 3:4]).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        return eo, sar, mask, fname

def get_dataloaders(config):
    image_size = config["data"]["image_size"]
    batch_size = config["training"]["batch_size"]

    train_ds = EOSARDataset(config["data"]["train_dir"], image_size, augment=True)
    val_ds   = EOSARDataset(config["data"]["val_dir"],   image_size, augment=False)
    test_ds  = EOSARDataset(config["data"]["test_dir"],  image_size, augment=False)

    # Better sampler: sqrt weighting avoids over-sampling rare tiles
    # while still upweighting change-containing images
    ratios  = np.array(train_ds.change_ratios)
    weights = np.where(ratios > 0, np.sqrt(ratios) + 0.05, 0.01)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=4, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    pos = (ratios > 0).sum()
    print(f"Train: {len(train_ds)} ({pos} with change) | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader
