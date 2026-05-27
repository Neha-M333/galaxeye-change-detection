# eval.py
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import json

from model import ChangeDetectionModel
from dataset import ChangeDetectionDataset


def tta_predict(model, eo, sar, device):
    """10-fold TTA: 4 rotations x 2 flips."""
    eo  = eo.to(device)
    sar = sar.to(device)
    preds = []
    for k in range(4):
        eo_r  = torch.rot90(eo,  k, dims=[2, 3])
        sar_r = torch.rot90(sar, k, dims=[2, 3])
        for flip in [False, True]:
            eo_f  = torch.flip(eo_r,  [3]) if flip else eo_r
            sar_f = torch.flip(sar_r, [3]) if flip else sar_r
            with torch.no_grad():
                pred = torch.sigmoid(model(eo_f, sar_f, training=False))
            if flip:
                pred = torch.flip(pred, [3])
            pred = torch.rot90(pred, -k, dims=[2, 3])
            preds.append(pred.cpu())
    return torch.stack(preds).mean(0)


def evaluate(model, loader, device, threshold=0.5, use_tta=True):
    all_probs, all_preds, all_targets = [], [], []

    for eo, sar, mask in loader:
        if use_tta:
            prob = tta_predict(model, eo, sar, device)
        else:
            eo, sar = eo.to(device), sar.to(device)
            with torch.no_grad():
                prob = torch.sigmoid(
                    model(eo, sar, training=False)).cpu()

        pred = (prob > threshold).float()
        all_probs.append(prob.numpy())
        all_preds.append(pred.numpy())
        all_targets.append(mask.numpy())

    all_probs   = np.concatenate(all_probs,   axis=0)
    all_preds   = np.concatenate(all_preds,   axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    p_flat = all_preds[:, 0].flatten().astype(int)
    t_flat = all_targets.flatten().astype(int)

    tp = ((p_flat == 1) & (t_flat == 1)).sum()
    fp = ((p_flat == 1) & (t_flat == 0)).sum()
    fn = ((p_flat == 0) & (t_flat == 1)).sum()
    tn = ((p_flat == 0) & (t_flat == 0)).sum()

    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)
    iou       = tp / (tp + fp + fn + 1e-6)

    return {
        "f1": float(f1), "iou": float(iou),
        "precision": float(precision), "recall": float(recall),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "probs": all_probs, "preds": all_preds, "targets": all_targets
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate EO-SAR change detection model")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to test data folder")
    parser.add_argument("--weights",   type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Prediction threshold (default: 0.5)")
    parser.add_argument("--no_tta",    action="store_true",
                        help="Disable test time augmentation")
    parser.add_argument("--batch_size",type=int, default=4,
                        help="Batch size (default: 4)")
    parser.add_argument("--output_dir",type=str, default="eval_output",
                        help="Directory to save output figures")
    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load dataset
    dataset = ChangeDetectionDataset(
        data_path=args.data_path,
        split="test"
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=2, pin_memory=True
    )

    # Load model
    model = ChangeDetectionModel().to(device)
    checkpoint = torch.load(args.weights, map_location=device,
                            weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Loaded checkpoint (epoch {checkpoint['epoch']}, "
          f"Val F1={checkpoint['val_f1']:.4f})")

    # Evaluate
    use_tta = not args.no_tta
    print(f"\nRunning evaluation (TTA={'ON' if use_tta else 'OFF'})...")
    results = evaluate(model, loader, device,
                       threshold=args.threshold, use_tta=use_tta)

    # Print metrics
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"F1 Score:  {results['f1']:.4f}")
    print(f"IoU:       {results['iou']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"Threshold: {args.threshold}")
    print(f"TTA:       {'ON' if use_tta else 'OFF'}")

    # Save metrics
    metrics_out = {
        "f1":        round(results["f1"],        4),
        "iou":       round(results["iou"],       4),
        "precision": round(results["precision"], 4),
        "recall":    round(results["recall"],    4),
        "threshold": args.threshold,
        "tta":       use_tta
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\nMetrics saved to {output_dir}/metrics.json")

    # Confusion matrix
    cm = np.array([[results["tn"], results["fp"]],
                   [results["fn"], results["tp"]]])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        cm, display_labels=["No Change", "Change"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix (Test Split)")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {output_dir}/confusion_matrix.png")

    # Prediction visualisations (6 samples)
    EO_MEAN = np.array([0.485, 0.456, 0.406])
    EO_STD  = np.array([0.229, 0.224, 0.225])
    SAR_MEAN, SAR_STD = 2.9777, 1.7104

    samples = list(dataset)
    indices = np.linspace(0, len(samples)-1, 6, dtype=int)

    fig, axes = plt.subplots(6, 4, figsize=(14, 21))
    axes[0, 0].set_title("EO (pre-event)")
    axes[0, 1].set_title("SAR (post-event)")
    axes[0, 2].set_title("Ground Truth")
    axes[0, 3].set_title("Prediction")

    for row, idx in enumerate(indices):
        eo, sar, mask = samples[idx]
        prob = results["probs"][idx, 0]
        pred = (prob > args.threshold).astype(float)

        eo_disp  = np.clip(
            eo.numpy().transpose(1,2,0) * EO_STD + EO_MEAN, 0, 1)
        sar_disp = np.clip(
            np.expm1(sar.numpy()[0] * SAR_STD + SAR_MEAN) / 238.0, 0, 1)

        axes[row, 0].imshow(eo_disp)
        axes[row, 1].imshow(sar_disp, cmap="gray")
        axes[row, 2].imshow(mask.numpy(), cmap="Reds", vmin=0, vmax=1)
        axes[row, 3].imshow(pred,         cmap="Reds", vmin=0, vmax=1)
        for col in range(4):
            axes[row, col].axis("off")

    plt.suptitle("Test Predictions", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "predictions.png", dpi=150)
    plt.close()
    print(f"Predictions saved to {output_dir}/predictions.png")
    print("\nDone ✅")


if __name__ == "__main__":
    main()
