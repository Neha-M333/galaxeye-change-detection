import os
import gc
import random
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from dataloader import get_dataloaders, load_config
from model import get_model, get_loss
from eval import compute_metrics

# Memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train EO-SAR Change Detection Model")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config YAML file")
    return parser.parse_args()


def main():
    args   = parse_args()
    config = load_config(args.config)

    set_seed(config["training"].get("random_seed", 42))

    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, _ = get_dataloaders(config)

    model   = get_model(config).to(device)
    loss_fn = get_loss(config).to(device)

    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )

    epochs    = config["training"]["epochs"]
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config["training"]["learning_rate"],
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=config["training"].get("warmup_epochs", 3) / epochs,
        anneal_strategy="cos",
        div_factor=25,
        final_div_factor=1e4
    )

    scaler = torch.amp.GradScaler("cuda")

    save_dir = config["checkpoint"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    best_f1    = 0.0
    best_epoch = 0
    patience   = 10
    no_improve = 0

    for epoch in range(1, epochs + 1):

        # ── Train ──────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for eo, sar, masks, _ in tqdm(train_loader,
                                       desc=f"Ep{epoch:02d} Train",
                                       leave=False):
            eo    = eo.to(device, non_blocking=True)
            sar   = sar.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                logits = model(eo, sar)
                loss   = loss_fn(logits, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config["training"]["gradient_clip"]
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()
            del eo, sar, masks, logits, loss

        train_loss /= len(train_loader)

        # ── Validate ───────────────────────────────────────────
        model.eval()
        val_loss  = 0.0
        all_preds = []
        all_masks = []

        with torch.no_grad():
            for eo, sar, masks, _ in tqdm(val_loader,
                                           desc=f"Ep{epoch:02d} Val",
                                           leave=False):
                eo  = eo.to(device, non_blocking=True)
                sar = sar.to(device, non_blocking=True)

                with torch.amp.autocast("cuda"):
                    logits = model(eo, sar)
                    loss   = loss_fn(logits, masks.to(device))

                val_loss += loss.item()
                preds = (torch.sigmoid(logits) > 0.5).float().cpu()
                all_preds.append(preds)
                all_masks.append(masks)
                del eo, sar, logits, loss

        val_loss  /= len(val_loader)
        all_preds  = torch.cat(all_preds)
        all_masks  = torch.cat(all_masks)
        metrics    = compute_metrics(all_preds, all_masks)

        print(f"Epoch {epoch:03d}/{epochs} | "
              f"TrLoss {train_loss:.4f} | VaLoss {val_loss:.4f} | "
              f"IoU {metrics['iou']:.4f} | F1 {metrics['f1']:.4f} | "
              f"P {metrics['precision']:.4f} | R {metrics['recall']:.4f} | "
              f"LR {scheduler.get_last_lr()[0]:.2e}")

        # ── Save best ──────────────────────────────────────────
        if metrics["f1"] > best_f1:
            best_f1    = metrics["f1"]
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_f1":  metrics["f1"],
                    "val_iou": metrics["iou"],
                    "config":  config
                },
                os.path.join(save_dir, "best_model.pth")
            )
            print(f"  *** New best! F1: {best_f1:.4f} | IoU: {metrics['iou']:.4f} ***")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

        # ── Periodic checkpoint ────────────────────────────────
        if epoch % 5 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_f1": metrics["f1"]
                },
                os.path.join(save_dir, f"checkpoint_epoch_{epoch:03d}.pth")
            )
            print(f"  Checkpoint saved: epoch {epoch}")

        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nDone! Best epoch: {best_epoch} | Best Val F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
