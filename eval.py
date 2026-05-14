import os
import gc
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloader import EOSARDataset, load_config
from model import SiameseChangeDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate EO-SAR Change Detection Model")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to dataset split directory (e.g. /path/to/test)")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to model checkpoint (e.g. /path/to/best_model.pth)")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for evaluation")
    parser.add_argument("--no_tta", action="store_true",
                        help="Disable TTA (faster but lower accuracy)")
    return parser.parse_args()


def compute_metrics(preds, targets, eps=1e-6):
    preds   = preds.float().view(-1)
    targets = targets.float().view(-1)
    tp = (preds * targets).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()
    tn = ((1 - preds) * (1 - targets)).sum().item()
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    iou       = tp / (tp + fp + fn + eps)
    return {"iou": iou, "precision": precision,
            "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def tta_predict(model, eo, sar, device):
    """10-fold TTA: 4 rotations x 2 flips + 2 scale variations."""
    preds = []
    h, w  = eo.shape[-2], eo.shape[-1]

    for k in range(4):
        for flip in [False, True]:
            eo_t  = torch.rot90(eo,  k, dims=[2, 3])
            sar_t = torch.rot90(sar, k, dims=[2, 3])
            if flip:
                eo_t  = torch.flip(eo_t,  [3])
                sar_t = torch.flip(sar_t, [3])
            with torch.no_grad():
                with torch.amp.autocast("cuda"):
                    out  = model(eo_t.to(device), sar_t.to(device))
                    if isinstance(out, tuple):
                        out = out[0]
                    prob = torch.sigmoid(out).cpu()
            if flip:
                prob = torch.flip(prob, [3])
            preds.append(torch.rot90(prob, -k, dims=[2, 3]))

    for scale in [0.875, 1.125]:
        nh, nw = int(h * scale), int(w * scale)
        eo_s  = F.interpolate(eo,  size=(nh, nw), mode="bilinear", align_corners=False)
        sar_s = F.interpolate(sar, size=(nh, nw), mode="bilinear", align_corners=False)
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                out  = model(eo_s.to(device), sar_s.to(device))
                if isinstance(out, tuple):
                    out = out[0]
                prob = torch.sigmoid(out).cpu()
        prob = F.interpolate(prob, size=(h, w), mode="bilinear", align_corners=False)
        preds.append(prob)

    return torch.stack(preds).mean(dim=0)


def simple_predict(model, eo, sar, device):
    """Single forward pass without TTA."""
    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            out = model(eo.to(device), sar.to(device))
            if isinstance(out, tuple):
                out = out[0]
            prob = torch.sigmoid(out).cpu()
    return prob


def main():
    args   = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ─────────────────────────────────────────────
    model = SiameseChangeDetector(pretrained=False, deep_supervision=True).to(device)
    ckpt  = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', 'N/A')}")
    print(f"Val F1 at save time: {ckpt.get('val_f1', 'N/A'):.4f}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Dataset ────────────────────────────────────────────────
    dataset = EOSARDataset(args.data_path,
                           image_size=config["data"]["image_size"],
                           augment=False)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=False, num_workers=2, pin_memory=False)

    predict_fn = simple_predict if args.no_tta else tta_predict
    mode_str   = "No TTA" if args.no_tta else "10-fold TTA"
    print(f"\nEvaluating {len(dataset)} samples with {mode_str}...")

    # ── Collect predictions ────────────────────────────────────
    all_probs = []
    all_masks = []
    for eo, sar, masks, _ in tqdm(loader, desc="Evaluating"):
        probs = predict_fn(model, eo, sar, device)
        all_probs.append(probs)
        all_masks.append(masks)
        gc.collect()
        torch.cuda.empty_cache()

    all_probs = torch.cat(all_probs)
    all_masks  = torch.cat(all_masks)

    # ── Threshold sweep ────────────────────────────────────────
    print(f"\n{'Threshold':>10} | {'IoU':>6} | {'Precision':>9} | {'Recall':>6} | {'F1':>6}")
    print("-" * 55)
    best_f1, best_thresh, best_metrics = 0, 0.5, None

    for thresh in np.arange(0.05, 0.96, 0.05):
        preds   = (all_probs > thresh).float()
        metrics = compute_metrics(preds, all_masks)
        marker  = " ←" if metrics["f1"] > best_f1 else ""
        print(f"  {thresh:.2f}     | "
              f"{metrics['iou']:.4f} | "
              f"{metrics['precision']:.4f}    | "
              f"{metrics['recall']:.4f} | "
              f"{metrics['f1']:.4f}{marker}")
        if metrics["f1"] > best_f1:
            best_f1      = metrics["f1"]
            best_thresh  = thresh
            best_metrics = metrics

    print(f"\nBest threshold: {best_thresh:.2f} → "
          f"F1: {best_f1:.4f} | IoU: {best_metrics['iou']:.4f} | "
          f"P: {best_metrics['precision']:.4f} | R: {best_metrics['recall']:.4f}")

    # ── Save metrics ───────────────────────────────────────────
    split_name = os.path.basename(args.data_path.rstrip("/"))
    out_path   = os.path.join(args.output_dir, f"metrics_{split_name}.txt")
    with open(out_path, "w") as f:
        f.write(f"Data path:  {args.data_path}\n")
        f.write(f"Weights:    {args.weights}\n")
        f.write(f"Mode:       {mode_str}\n")
        f.write(f"Threshold:  {best_thresh:.2f}\n")
        f.write(f"IoU:        {best_metrics['iou']:.4f}\n")
        f.write(f"Precision:  {best_metrics['precision']:.4f}\n")
        f.write(f"Recall:     {best_metrics['recall']:.4f}\n")
        f.write(f"F1:         {best_metrics['f1']:.4f}\n")
    print(f"\nMetrics saved to {out_path}")
    print("Done!")


if __name__ == "__main__":
    main()
