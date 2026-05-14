
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

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

def plot_confusion_matrix(metrics, split_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    tp, fp = metrics["tp"], metrics["fp"]
    fn, tn = metrics["fn"], metrics["tn"]
    total  = tp + fp + fn + tn
    cm      = np.array([[tn, fp], [fn, tp]])
    cm_norm = cm / total
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im)
    labels = ["No-Change (0)", "Change (1)"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {split_name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i,
                    f"{cm[i,j]:.0f}\n({cm_norm[i,j]:.2%})",
                    ha="center", va="center",
                    color="white" if cm_norm[i,j] > 0.5 else "black")
    plt.tight_layout()
    path = os.path.join(save_dir, f"confusion_matrix_{split_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

def plot_predictions(model, dataset, device,
                     save_dir, split_name, n=5):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    for i in range(min(n, len(dataset))):
        eo, sar, mask, fname = dataset[i]
        with torch.no_grad():
            logit = model(eo.unsqueeze(0).to(device),
                          sar.unsqueeze(0).to(device))
            pred  = (torch.sigmoid(logit) > 0.5).float().cpu().squeeze()
        pre_img  = eo.permute(1, 2, 0).numpy()
        post_img = sar.squeeze().numpy()
        gt_mask  = mask.squeeze().numpy()
        pr_mask  = pred.numpy()
        pre_img  = (pre_img - pre_img.min()) / (
                    pre_img.max() - pre_img.min() + 1e-6)
        post_img = (post_img - post_img.min()) / (
                    post_img.max() - post_img.min() + 1e-6)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(pre_img)
        axes[0].set_title("Pre-Event (EO)")
        axes[1].imshow(post_img, cmap="gray")
        axes[1].set_title("Post-Event (SAR)")
        axes[2].imshow(gt_mask, cmap="Reds")
        axes[2].set_title("Ground Truth")
        axes[3].imshow(pr_mask, cmap="Reds")
        axes[3].set_title("Prediction")
        for ax in axes:
            ax.axis("off")
        plt.suptitle(f"Example {i+1} — {fname}", fontsize=10)
        plt.tight_layout()
        path = os.path.join(save_dir,
                            f"{split_name}_example_{i+1}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")
