# Binary Change Detection on EO-SAR Image Pairs

Given a pre-event RGB (EO) image and a post-event SAR image, this model predicts a binary pixel-level mask indicating changed (damaged/destroyed) vs unchanged regions. The architecture uses two parallel EfficientNet-B4 encoders fused at each scale via a gated CrossModalFusion module, decoded with CBAM/SCSE attention, and trained with a combined Focal + Dice + Tversky loss to handle severe class imbalance.

---

## Requirements

Python 3.10. Save the following as `requirements.txt`:

```
torch==2.1.0
torchvision==0.16.0
timm==0.9.12
rasterio==1.3.9
albumentations==1.3.1
segmentation-models-pytorch==0.3.3
pyyaml==6.0.1
numpy==1.24.3
matplotlib==3.7.2
tqdm==4.66.1
```

---

## Environment Setup

```bash
conda create -n galaxeye python=3.10
conda activate galaxeye
pip install -r requirements.txt
```

---

## Dataset Structure

```
galaxeye-dataset/
├── train/
│   └── train/
│       ├── pre-event/
│       ├── post-event/
│       └── target/
├── val/
│   └── val/
│       ├── pre-event/
│       ├── post-event/
│       └── target/
└── test/
    └── test/
        ├── pre-event/
        ├── post-event/
        └── target/
```

Update paths in `config.yaml` to match your local layout.

---

## Training

```bash
python train.py --config config.yaml --data_dir /path/to/galaxeye-dataset
```

---

## Evaluation

```bash
python eval.py --data_path /path/to/test \
               --weights   /path/to/best_model.pth
```

---

## Model Weights

[Download best_model.pth (~150 MB) — Google Drive](https://drive.google.com/file/d/17pHEZLWeh0milq3EPz2ExbzcVvZzkYCO/view?usp=sharing)

---

## Results

| Split | Threshold | IoU    | Precision | Recall | F1     |
|-------|-----------|--------|-----------|--------|--------|
| Val   | 0.70      | 0.5774 | 0.6384    | 0.8581 | 0.7321 |
| Test  | 0.10      | 0.0303 | 0.0971    | 0.0422 | 0.0588 |
| Test (change tiles only, 60 tiles) | 0.10 | 0.0303 | 0.0971 | 0.0422 | 0.0588 |

> **Note:** Test set (scene_09, scene_10) represents unseen disaster events. Diagnostic analysis confirmed severe domain shift — model confidence on true change pixels collapsed from mean prob 0.8286 (val) to 0.0678 (test). Only 17 of 77 test tiles contain zero change pixels. Root cause is scene-level domain shift — test tiles originate from entirely unseen disaster events not represented in training. Full analysis in the technical report.

---

## Citation / References

- Daudt et al. (2018). *Fully Convolutional Siamese Networks for Change Detection.* ICIP.
- Chen & Shi (2021). *Remote Sensing Image Change Detection with Transformers.* IEEE TGRS.
- Fang et al. (2021). *SNUNet-CD: A Densely Connected Siamese Network for Change Detection.* IEEE GRSL.
- Bandara & Patel (2022). *A Transformer-Based Siamese Network for Change Detection.* IGARSS.
- Hughes & Schmitt (2018). *Mining Hard Negative Samples for SAR-Optical Data Matching.* Remote Sensing.
- Schmitt & Zhu (2016). *Data Fusion and Remote Sensing.* IEEE GRSM.
- Ronneberger et al. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI.
- Tan & Le (2019). *EfficientNet: Rethinking Model Scaling for CNNs.* ICML.
- Woo et al. (2018). *CBAM: Convolutional Block Attention Module.* ECCV.
- Roy et al. (2018). *Concurrent Spatial and Channel Squeeze & Excitation in FCNs.* MICCAI.
- Lin et al. (2017). *Focal Loss for Dense Object Detection.* ICCV.
- Milletari et al. (2016). *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation.* 3DV.
- Salehi et al. (2017). *Tversky Loss Function for Image Segmentation.* MICCAI Workshop.

---

*Neha M — GalaxEye Space AI Research Intern Technical Assignment*
