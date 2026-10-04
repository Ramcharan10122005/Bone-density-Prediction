"""Two-stage 3D UNet medical imaging segmentation pipeline for CBCT implant site analysis.

Faithfully extracted from the source Kaggle notebook (lol-idk.ipynb).

Stage 1: Jaw Region of Interest (ROI) segmentation via sliding window inference.
Stage 2: Tight implant segmentation within the cropped and resized (96^3) ROI.
Bone Quality: Classification (D1-D4) based on mean normalized intensity in the implant mask.
3D Mesh Generation: Marching cubes for ROI and implant surfaces.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import ndimage
from skimage.measure import marching_cubes
import torch

from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    Resize,
    ScaleIntensityRanged,
    SpatialPadd,
    Spacingd,
)

# -------------------------------------------------------------------------
# Global Configuration & Defaults matching Kaggle notebook
# -------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
STAGE1_MODEL_PATH = MODELS_DIR / "best_stage1_model.pth"
STAGE2_MODEL_PATH = MODELS_DIR / "best_stage2_model.pth"

# Device selection: CUDA -> CPU (avoiding MPS due to PyTorch missing slow_conv3d_forward)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Bone Quality classification thresholds
# Cell 105 (exact table in notebook screenshot):
D1_THRESHOLD = 0.6429
D2_THRESHOLD = 0.5286
D3_THRESHOLD = 0.3857

# Alternative Cell 111 thresholds:
D1_ALT = 0.75
D2_ALT = 0.50
D3_ALT = 0.35


# -------------------------------------------------------------------------
# Model Definitions
# -------------------------------------------------------------------------
def create_stage1_model() -> UNet:
    """Stage-1 UNet: Jaw ROI segmentation.

    spatial_dims=3, in_channels=1, out_channels=2,
    channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2,
    norm="INSTANCE", dropout=0.2
    """
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="INSTANCE",
        dropout=0.2,
    )


def create_stage2_model() -> UNet:
    """Stage-2 UNet: Implant segmentation.

    spatial_dims=3, in_channels=1, out_channels=2,
    channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2), num_res_units=2,
    norm="INSTANCE", dropout=0.2
    """
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="INSTANCE",
        dropout=0.2,
    )


class PipelineModelManager:
    """Cached loader for Stage 1 and Stage 2 models."""

    _stage1_model: Optional[UNet] = None
    _stage2_model: Optional[UNet] = None

    @classmethod
    def get_models(
        cls,
        stage1_path: Path = STAGE1_MODEL_PATH,
        stage2_path: Path = STAGE2_MODEL_PATH,
        device: torch.device = DEVICE,
    ) -> Tuple[UNet, UNet]:
        if cls._stage1_model is None:
            if not stage1_path.exists():
                raise FileNotFoundError(f"Stage 1 weights not found at: {stage1_path}")
            model1 = create_stage1_model().to(device)
            state_dict1 = torch.load(stage1_path, map_location=device, weights_only=False)
            model1.load_state_dict(state_dict1)
            model1.eval()
            cls._stage1_model = model1

        if cls._stage2_model is None:
            if not stage2_path.exists():
                raise FileNotFoundError(f"Stage 2 weights not found at: {stage2_path}")
            model2 = create_stage2_model().to(device)
            state_dict2 = torch.load(stage2_path, map_location=device, weights_only=False)
            model2.load_state_dict(state_dict2)
            model2.eval()
            cls._stage2_model = model2

        return cls._stage1_model, cls._stage2_model


# -------------------------------------------------------------------------
# Preprocessing Transforms
# -------------------------------------------------------------------------
def get_inference_transforms() -> Compose:
    """Exact preprocessing transforms for inference on input CBCT volume:

    - LoadImaged
    - EnsureChannelFirstd
    - Orientationd(axcodes="RAS")
    - Spacingd(pixdim=(0.8, 0.8, 0.8), mode="bilinear")
    - ScaleIntensityRanged(a_min=-1000, a_max=2500, b_min=0.0, b_max=1.0, clip=True)
    - SpatialPadd(spatial_size=(192, 144, 128))
    - EnsureTyped
    """
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=(0.8, 0.8, 0.8),
            mode="bilinear",
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1000,
            a_max=2500,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        SpatialPadd(
            keys=["image"],
            spatial_size=(192, 144, 128),
        ),
        EnsureTyped(keys=["image"]),
    ])


def get_evaluation_transforms() -> Compose:
    """Exact preprocessing transforms for volume + ground truth segmentation:

    - LoadImaged(keys=["image", "label"])
    - EnsureChannelFirstd
    - Orientationd(axcodes="RAS")
    - Spacingd(pixdim=(0.8, 0.8, 0.8), mode=("bilinear", "nearest"))
    - ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=2500, b_min=0.0, b_max=1.0, clip=True)
    - SpatialPadd(keys=["image", "label"], spatial_size=(192, 144, 128))
    - EnsureTyped(keys=["image", "label"])
    """
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(0.8, 0.8, 0.8),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1000,
            a_max=2500,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        SpatialPadd(
            keys=["image", "label"],
            spatial_size=(192, 144, 128),
        ),
        EnsureTyped(keys=["image", "label"]),
    ])


# -------------------------------------------------------------------------
# ROI Cropping & Connected Component Analysis (Cell 54 from notebook)
# -------------------------------------------------------------------------
def extract_roi_crop(
    image_np: np.ndarray,
    roi_mask: np.ndarray,
    margin: int = 20,
    return_mask: bool = False,
) -> Any:
    """Extract crop around largest connected component of ROI mask with margin.

    Reproduces Cell 54 in lol-idk.ipynb:
    - ndimage.label on mask
    - Keep largest connected component by ndimage.sum
    - Bounding box: coords.min(axis=0) to coords.max(axis=0)
    - Margin: max(0, min - margin), min(dim, max + margin)
    - Return cropped image and bounding box coordinates (zmin, ymin, xmin, zmax, ymax, xmax)
    """
    binary_mask = (roi_mask > 0).astype(np.uint8)
    labels, num = ndimage.label(binary_mask)

    if num == 0:
        # Fallback if no component found: center crop of 96^3 or full volume
        d, h, w = image_np.shape
        zmin = max(0, (d - 96) // 2)
        ymin = max(0, (h - 96) // 2)
        xmin = max(0, (w - 96) // 2)
        zmax = min(d, zmin + 96)
        ymax = min(h, ymin + 96)
        xmax = min(w, xmin + 96)
        crop = image_np[zmin:zmax, ymin:ymax, xmin:xmax]
        bbox = (zmin, ymin, xmin, zmax, ymax, xmax)
        return (crop, bbox, binary_mask) if return_mask else (crop, bbox)

    sizes = ndimage.sum(binary_mask, labels, range(1, num + 1))
    largest = int(np.argmax(sizes)) + 1
    largest_component = (labels == largest)

    coords = np.argwhere(largest_component)
    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0)

    # Margin expansion
    zmin = int(max(0, zmin - margin))
    ymin = int(max(0, ymin - margin))
    xmin = int(max(0, xmin - margin))

    zmax = int(min(image_np.shape[0], zmax + margin))
    ymax = int(min(image_np.shape[1], ymax + margin))
    xmax = int(min(image_np.shape[2], xmax + margin))

    crop = image_np[zmin:zmax, ymin:ymax, xmin:xmax]
    bbox = (zmin, ymin, xmin, zmax, ymax, xmax)
    return (crop, bbox, largest_component) if return_mask else (crop, bbox)


# -------------------------------------------------------------------------
# Bone Quality Classification (Cells 111 & 112 from notebook)
# -------------------------------------------------------------------------
def classify_bone_quality(mean_val: float, mode: str = "cell105") -> str:
    """Bone quality classification from lol-idk.ipynb:

    Mode "cell105" (matches the notebook's displayed 15-case table):
        mean_val > 0.6429 -> D1
        mean_val > 0.5286 -> D2
        mean_val > 0.3857 -> D3
        else              -> D4

    Mode "cell111":
        mean_val >= 0.75  -> D1
        mean_val >= 0.50  -> D2
        mean_val >= 0.35  -> D3
        else              -> D4
    """
    if mode == "cell105":
        if mean_val > D1_THRESHOLD:
            return "D1"
        elif mean_val > D2_THRESHOLD:
            return "D2"
        elif mean_val > D3_THRESHOLD:
            return "D3"
        else:
            return "D4"
    else:
        if mean_val >= D1_ALT:
            return "D1"
        elif mean_val >= D2_ALT:
            return "D2"
        elif mean_val >= D3_ALT:
            return "D3"
        else:
            return "D4"


# -------------------------------------------------------------------------
# Per-case Metric Calculations (Cells 109 & 120 from notebook)
# -------------------------------------------------------------------------
def calculate_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
    eps: float = 1e-8,
) -> Dict[str, Any]:
    """Calculate per-case segmentation metrics matching notebook Cells 109 and 120:

    Dice, IoU, Precision, Recall, Specificity, Accuracy.
    """
    gt_bin = (gt == 1).astype(bool)
    pred_bin = (pred == 1).astype(bool)

    tp = float(np.logical_and(gt_bin, pred_bin).sum())
    fp = float(np.logical_and(~gt_bin, pred_bin).sum())
    fn = float(np.logical_and(gt_bin, ~pred_bin).sum())
    tn = float(np.logical_and(~gt_bin, ~pred_bin).sum())

    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    gt_volume = int(gt_bin.sum())
    pred_volume = int(pred_bin.sum())
    diff = pred_volume - gt_volume
    error_pct = (diff / gt_volume * 100.0) if gt_volume > 0 else 0.0

    return {
        "dice": float(round(dice, 4)),
        "iou": float(round(iou, 4)),
        "precision": float(round(precision, 4)),
        "recall": float(round(recall, 4)),
        "specificity": float(round(specificity, 4)),
        "accuracy": float(round(accuracy, 4)),
        "gt_volume": gt_volume,
        "pred_volume": pred_volume,
        "difference": diff,
        "error_percent": float(round(error_pct, 2)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


# -------------------------------------------------------------------------
# Mesh Extraction (marching_cubes)
# -------------------------------------------------------------------------
def extract_mesh(
    binary_mask: np.ndarray,
    level: float = 0.5,
    offset: Optional[Tuple[int, int, int]] = None,
    scale: Optional[Tuple[float, float, float]] = None,
) -> Dict[str, list]:
    """Extract surface mesh using skimage.measure.marching_cubes.

    Returns vertices and faces as lists for JSON serialization.
    Optionally scales and offsets vertices to align crop coordinates with full volume.
    """
    if np.sum(binary_mask > 0) < 4:
        return {"vertices": [], "faces": []}

    try:
        verts, faces, _, _ = marching_cubes(binary_mask.astype(float), level=level)

        if scale is not None:
            verts[:, 0] *= scale[0]
            verts[:, 1] *= scale[1]
            verts[:, 2] *= scale[2]

        if offset is not None:
            verts[:, 0] += offset[0]
            verts[:, 1] += offset[1]
            verts[:, 2] += offset[2]

        return {
            "vertices": np.round(verts, 2).tolist(),
            "faces": faces.astype(int).tolist(),
        }
    except Exception as err:
        print(f"Warning: marching_cubes failed: {err}")
        return {"vertices": [], "faces": []}


# -------------------------------------------------------------------------
# Main Execution Pipeline
# -------------------------------------------------------------------------
def run_pipeline(
    nrrd_path: str | Path,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Execute the full two-stage segmentation and bone-quality pipeline on a CBCT NRRD file.

    Parameters:
        nrrd_path: Absolute or relative path to the .nrrd volume.
        device: PyTorch device (defaults to CUDA/MPS/CPU).

    Returns:
        {
          "bone_quality": str,                 # "D1", "D2", "D3", or "D4"
          "implant_volume_voxels": int,        # Total implant voxels predicted
          "roi_mesh": {"vertices": [...], "faces": [...]},
          "implant_mesh": {"vertices": [...], "faces": [...]}
        }
    """
    nrrd_path = str(Path(nrrd_path).resolve())
    if not os.path.exists(nrrd_path):
        raise FileNotFoundError(f"NRRD file not found: {nrrd_path}")

    dev = device or DEVICE
    stage1_model, stage2_model = PipelineModelManager.get_models(device=dev)

    # 1. Preprocess full volume
    transforms = get_inference_transforms()
    batch_data = transforms({"image": nrrd_path})
    image_tensor = batch_data["image"].unsqueeze(0).to(dev)  # [1, 1, D, H, W]
    image_np = image_tensor[0, 0].detach().cpu().numpy()     # [D, H, W]

    # 2. Stage-1 Jaw ROI segmentation via sliding window inference
    with torch.no_grad():
        stage1_outputs = sliding_window_inference(
            image_tensor,
            roi_size=(192, 144, 128),
            sw_batch_size=1,
            predictor=stage1_model,
            overlap=0.5,
        )
        stage1_pred = torch.argmax(stage1_outputs, dim=1).detach().cpu().numpy().squeeze(0).astype(np.uint8)

    # 3. Extract ROI crop around largest connected component
    image_crop, bbox, largest_component = extract_roi_crop(
        image_np, stage1_pred, margin=20, return_mask=True
    )
    zmin, ymin, xmin, zmax, ymax, xmax = bbox

    # 4. Resize crop to 96^3 (exact match to Resize(spatial_size=(96, 96, 96), mode="nearest"))
    resize_transform = Resize(spatial_size=(96, 96, 96), mode="nearest")
    crop_tensor = torch.tensor(image_crop, dtype=torch.float32).unsqueeze(0)  # [1, D_c, H_c, W_c]
    crop_resized = resize_transform(crop_tensor)                             # [1, 96, 96, 96]
    stage2_input = crop_resized.unsqueeze(0).to(dev)                         # [1, 1, 96, 96, 96]

    # 5. Stage-2 Implant segmentation within 96^3 crop
    with torch.no_grad():
        stage2_outputs = stage2_model(stage2_input)
        stage2_pred = torch.argmax(stage2_outputs, dim=1).detach().cpu().numpy().squeeze(0).astype(np.uint8)

    # 6. Bone Quality Classification (Cells 110-112)
    crop_resized_np = crop_resized[0].detach().cpu().numpy()
    implant_mask = (stage2_pred == 1)
    implant_volume_voxels = int(np.sum(implant_mask))

    if implant_volume_voxels > 0:
        mean_intensity = float(np.mean(crop_resized_np[implant_mask]))
    else:
        mean_intensity = 0.0

    bone_quality = classify_bone_quality(mean_intensity)

    # 7. Surface Mesh Extraction using marching_cubes
    # ROI mesh in full volume space (isolated to the primary surgical alveolar bone bed)
    roi_mesh = extract_mesh(largest_component, level=0.5)

    # Implant mesh mapped back to full-volume coordinates so it aligns inside ROI
    crop_d = max(1, zmax - zmin)
    crop_h = max(1, ymax - ymin)
    crop_w = max(1, xmax - xmin)
    scale = (crop_d / 96.0, crop_h / 96.0, crop_w / 96.0)
    offset = (zmin, ymin, xmin)

    implant_mesh = extract_mesh(
        stage2_pred,
        level=0.5,
        offset=offset,
        scale=scale,
    )

    return {
        "bone_quality": bone_quality,
        "implant_volume_voxels": implant_volume_voxels,
        "roi_mesh": roi_mesh,
        "implant_mesh": implant_mesh,
        # Extended metadata (useful for inspection / API responses)
        "mean_intensity": round(mean_intensity, 4),
        "crop_bbox": {
            "zmin": zmin,
            "ymin": ymin,
            "xmin": xmin,
            "zmax": zmax,
            "ymax": ymax,
            "xmax": xmax,
        },
        "volume_shape": list(image_np.shape),
    }
