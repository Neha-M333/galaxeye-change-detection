import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path

from model import ChangeDetectionModel
from dataset import ChangeDetectionDataset, make_sampler


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        target = target.unsqueeze(1)
        bce    = F.binary_cross_entropy_with_logits(
            pred, target, reduction="none")
        prob   = torch.sigmoid(pred)
        p_t    = target * prob + (1 - target) * (1 - prob)
        return (self.alpha * (1 - p_t) ** self.gamma * bce).mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        target = target.unsqueeze(1)
        prob   = torch.sigmoid(pred)
        num    = 2 * (prob * target).sum() + self.smooth
        den    = prob.sum() + target.sum() + self.smooth
        return torch.clamp(1 - num / den, min=0.0)


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.smooth = smooth

    def forward(self, pred, target):
        target = target.unsqueeze(1)
        prob   = torch.sigmoid(pred)
        tp     = (prob * target).sum()
        fp     = (prob * (1 - target)).sum()
        fn     = ((1 - prob) * target).sum()
        return torch.clamp(
            1 - (tp + self.smooth) /
            (tp + self.alpha*fp + self.beta*fn + self.smooth),
            min=0.0)


class CompositeLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.focal   = FocalLoss(cfg["focal_alpha"], cfg["focal_gamma"])
        self.dice    = DiceLoss()
        self.tversky = TverskyLoss(
            cfg["tversky_alpha"], cfg["tversky_beta"])
        self.fw = cfg["focal_weight"]
        self.dw = cfg["dice_weight"]
        self.tw = cfg["tversky_weight"]

    def forward(self, pred, target):
        return (self.fw * self.focal(pred, target) +
                self.dw * self.dice(pred, target)  +
                self.tw * self.tversky(pred, target))


def compute_metrics(pred_logits, target, threshold=0.5):
    pred   = (torch.sigmoid(pred_logits) > threshold).float()
    target = target.unsqueeze(1)
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)
    iou       = tp / (tp + fp + fn + 1e-6)
    return f1, iou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, default="config.yaml")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root dataset directory")
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Datasets
    train_ds = ChangeDetectionDataset(
        Path(args.data_dir) / "train", split="train",
        image_size=cfg["data"]["image_size"])
    val_ds = ChangeDetectionDataset(
        Path(args.data_dir) / "val", split="val",
        image_size=cfg["data"]["image_size"])

    sampler = make_sampler(train_ds)
    train_loader = DataLoader(
        train_ds, batch_size=cfg["data"]["batch_size"],
        sampler=sampler, num_workers=cfg["data"]["num_workers"],
        pin_memory=True)
    val_loader = DataLoader(
        val_ds, batch_size=cfg["data"]["batch_size"],
        shuffle=False, num_workers=cfg["data"]["num_workers"],
        pin_memory=True)

    # Model
    model    = ChangeDetectionModel().to(device)
    loss_fn  = CompositeLoss(cfg["loss"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg["training"]["max_lr"],
        steps_per_epoch=len(train_loader),
        epochs=cfg["training"]["epochs"],
        pct_start=cfg["training"]["warmup_pct"])
    scaler = GradScaler()

    best_f1    = 0.0
    no_improve = 0
    history    = {"train_loss": [], "val_loss": [],
                  "val_f1": [], "val_iou": []}

    print(f"\nStarting training — {cfg['training']['epochs']} epochs")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>9} | "
          f"{'Val F1':>7} | {'Val IoU':>8} | {'Time':>6}")
    print("-" * 62)

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        t0 = time.time()
        model.train()
        train_losses = []

        for eo, sar, mask in train_loader:
            eo, sar, mask = eo.to(device), sar.to(device), mask.to(device)
            optimizer.zero_grad()
            with autocast():
                out, aux8, aux16 = model(eo, sar, training=True)
                loss = (loss_fn(out, mask)
                    + cfg["loss"]["aux_loss_weight"] * loss_fn(
                        F.interpolate(aux8, size=(256,256),
                                      mode="bilinear",
                                      align_corners=False), mask)
                    + cfg["loss"]["aux_loss_weight"] * loss_fn(
                        F.interpolate(aux16, size=(256,256),
                                      mode="bilinear",
                                      align_corners=False), mask))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg["training"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses, val_f1s, val_ious = [], [], []
        with torch.no_grad():
            for eo, sar, mask in val_loader:
                eo, sar, mask = (eo.to(device), sar.to(device),
                                 mask.to(device))
                out  = model(eo, sar, training=False)
                loss = loss_fn(out, mask)
                val_losses.append(loss.item())
                f1, iou = compute_metrics(out, mask)
                val_f1s.append(f1)
                val_ious.append(iou)

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        val_f1     = np.mean(val_f1s)
        val_iou    = np.mean(val_ious)
        elapsed    = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["val_iou"].append(val_iou)

        print(f"{epoch:>6} | {train_loss:>10.4f} | {val_loss:>9.4f} | "
              f"{val_f1:>7.4f} | {val_iou:>8.4f} | {elapsed:>5.1f}s")

        if val_f1 > best_f1:
            best_f1    = val_f1
            no_improve = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_f1":      val_f1,
                "val_iou":     val_iou,
            }, output_dir / "best_model.pth")
            print(f"         ✅ Best saved (F1={val_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= cfg["training"]["patience"]:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print(f"\nBest Val F1: {best_f1:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(history["val_f1"],  color="green")
    axes[1].set_title("Val F1")
    axes[2].plot(history["val_iou"], color="orange")
    axes[2].set_title("Val IoU")
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150)

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f)

    print(f"Outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
