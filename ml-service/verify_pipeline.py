"""Verification script for ML pipeline extraction.

Runs Stage 1 (ROI) and Stage 2 (Implant) inference against sample .nrrd and .seg.nrrd files
placed in ml-service/sample_data/, computing and printing:
- Stage-1 ROI Dice and IoU against label == 6
- Stage-2 Implant Dice and IoU against label == 7
- Precision, Recall, Specificity, Accuracy, and Volume comparisons
- Bone quality classification (D1-D4) comparison

Matches the metric calculations in lol-idk.ipynb Cells 109, 112, 120, 126, 129.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import Resize

# Ensure ml-service root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pipeline import (
    DEVICE,
    PipelineModelManager,
    calculate_metrics,
    classify_bone_quality,
    extract_roi_crop,
    get_evaluation_transforms,
    run_pipeline,
)


def find_sample_pairs(data_dir: Path) -> List[Tuple[Path, Path]]:
    """Scan directory for matching (.nrrd, .seg.nrrd) pairs.

    Recognizes:
    - Pair in same folder: image.nrrd and image.seg.nrrd
    - Pair with _Segmentation suffix: Reformatted Volume.nrrd and Reformatted Volume_Segmentation.seg.nrrd
    - Separate subdirectories with image and label files
    """
    pairs: List[Tuple[Path, Path]] = []
    if not data_dir.exists():
        return pairs

    all_files = list(data_dir.rglob("*.nrrd"))
    seg_files = [f for f in all_files if f.name.endswith(".seg.nrrd") or "seg" in f.name.lower()]
    img_files = [f for f in all_files if f not in seg_files and not f.name.endswith(".seg.nrrd")]

    # 1. Exact or suffix matching
    for img in img_files:
        stem = img.stem.replace(".seg", "")
        # Look for matching seg file
        matching_seg = None
        for seg in seg_files:
            if seg.parent == img.parent:
                matching_seg = seg
                break
            if stem in seg.stem or seg.stem.startswith(stem):
                matching_seg = seg
                break

        if matching_seg:
            pairs.append((img, matching_seg))
            seg_files.remove(matching_seg)

    # 2. If 1 img and 1 seg remaining, pair them
    if len(img_files) == 1 and len(seg_files) == 1 and len(pairs) == 0:
        pairs.append((img_files[0], seg_files[0]))

    return pairs


def verify_single_case(
    image_path: Path,
    label_path: Path,
    case_idx: int = 1,
    device: torch.device = DEVICE,
) -> Dict[str, any]:
    """Run full verification on one image and ground truth label pair."""
    print("=" * 70)
    print(f"VERIFYING CASE {case_idx}")
    print(f"Image : {image_path.name}")
    print(f"Label : {label_path.name}")
    print("=" * 70)

    stage1_model, stage2_model = PipelineModelManager.get_models(device=device)

    # 1. Preprocess with evaluation transforms
    transforms = get_evaluation_transforms()
    data = transforms({"image": str(image_path), "label": str(label_path)})

    image_tensor = data["image"].unsqueeze(0).to(device)  # [1, 1, D, H, W]
    label_tensor = data["label"].unsqueeze(0).to(device)  # [1, 1, D, H, W]

    image_np = image_tensor[0, 0].detach().cpu().numpy()
    raw_label_np = label_tensor[0, 0].detach().cpu().numpy()

    # Ground truth masks
    # Notebook convention: label == 6 is Jaw ROI, label == 7 is Implant
    if np.any(raw_label_np == 6):
        gt_roi = (raw_label_np == 6).astype(np.uint8)
    else:
        # Fallback if label is already binary ROI
        gt_roi = (raw_label_np == 1).astype(np.uint8)

    if np.any(raw_label_np == 7):
        gt_implant = (raw_label_np == 7).astype(np.uint8)
    else:
        gt_implant = (raw_label_np == 2).astype(np.uint8) if np.any(raw_label_np == 2) else np.zeros_like(gt_roi)

    # -------------------------------------------------------------
    # STAGE 1: Jaw ROI Inference
    # -------------------------------------------------------------
    print("\n[Stage 1] Running Sliding Window Inference for Jaw ROI...")
    with torch.no_grad():
        s1_outputs = sliding_window_inference(
            image_tensor,
            roi_size=(192, 144, 128),
            sw_batch_size=1,
            predictor=stage1_model,
            overlap=0.5,
        )
        stage1_pred = torch.argmax(s1_outputs, dim=1).detach().cpu().numpy().squeeze(0).astype(np.uint8)

    s1_metrics = calculate_metrics(gt_roi, stage1_pred)

    print("-" * 65)
    print("Stage-1 Jaw ROI Evaluation Metrics (vs label == 6):")
    print("-" * 65)
    print(f"Dice        : {s1_metrics['dice']:.4f}  (Notebook Validation: 0.7775 | Test: 0.8491)")
    print(f"IoU         : {s1_metrics['iou']:.4f}  (Notebook Validation: 0.6933 | Test: 0.7715)")
    print(f"Precision   : {s1_metrics['precision']:.4f}")
    print(f"Recall      : {s1_metrics['recall']:.4f}")
    print(f"Specificity : {s1_metrics['specificity']:.4f}")
    print(f"Accuracy    : {s1_metrics['accuracy']:.4f}")
    print(f"GT Volume   : {s1_metrics['gt_volume']} voxels")
    print(f"Pred Volume : {s1_metrics['pred_volume']} voxels")
    print(f"Volume Diff : {s1_metrics['difference']} ({s1_metrics['error_percent']}%)")

    # -------------------------------------------------------------
    # STAGE 2: Crop Extraction & Implant Inference
    # -------------------------------------------------------------
    print("\n[Stage 2] Running Stage-2 UNet (96^3) on Candidate Crop...")

    # (A) End-to-End: Crop based on Stage-1 predicted ROI
    image_crop_p1, bbox_p1 = extract_roi_crop(image_np, stage1_pred, margin=20)
    zmin_p1, ymin_p1, xmin_p1, zmax_p1, ymax_p1, xmax_p1 = bbox_p1

    # Ground truth implant inside predicted ROI crop
    gt_imp_in_p1 = gt_implant[zmin_p1:zmax_p1, ymin_p1:ymax_p1, xmin_p1:xmax_p1]

    resize_transform = Resize(spatial_size=(96, 96, 96), mode="nearest")

    crop_tensor_p1 = torch.tensor(image_crop_p1, dtype=torch.float32).unsqueeze(0)
    crop_resized_p1 = resize_transform(crop_tensor_p1)
    s2_input_p1 = crop_resized_p1.unsqueeze(0).to(device)

    gt_tensor_p1 = torch.tensor(gt_imp_in_p1, dtype=torch.float32).unsqueeze(0)
    gt_imp_resized_p1 = resize_transform(gt_tensor_p1)[0].numpy().astype(np.uint8)

    with torch.no_grad():
        s2_out_p1 = stage2_model(s2_input_p1)
        stage2_pred_p1 = torch.argmax(s2_out_p1, dim=1).detach().cpu().numpy().squeeze(0).astype(np.uint8)

    s2_metrics_e2e = calculate_metrics(gt_imp_resized_p1, stage2_pred_p1)

    # (B) Isolated Benchmark (Cell 54 / Cell 109): Crop centered on GT Implant
    has_gt_implant = np.any(gt_implant > 0)
    s2_metrics_isolated = None
    if has_gt_implant:
        # Largest connected component of GT implant
        lbl_imp, n_imp = ndimage.label(gt_implant)
        sz_imp = ndimage.sum(gt_implant, lbl_imp, range(1, n_imp + 1))
        largest_imp = int(np.argmax(sz_imp)) + 1
        imp_mask = (lbl_imp == largest_imp)

        coords_imp = np.argwhere(imp_mask)
        izmin, iymin, ixmin = coords_imp.min(axis=0)
        izmax, iymax, ixmax = coords_imp.max(axis=0)

        izmin = max(0, izmin - 20)
        iymin = max(0, iymin - 20)
        ixmin = max(0, ixmin - 20)
        izmax = min(image_np.shape[0], izmax + 20)
        iymax = min(image_np.shape[1], iymax + 20)
        ixmax = min(image_np.shape[2], ixmax + 20)

        image_crop_gt = image_np[izmin:izmax, iymin:iymax, ixmin:ixmax]
        gt_imp_crop_raw = gt_implant[izmin:izmax, iymin:iymax, ixmin:ixmax].astype(np.uint8)

        crop_tensor_gt = torch.tensor(image_crop_gt, dtype=torch.float32).unsqueeze(0)
        crop_resized_gt = resize_transform(crop_tensor_gt)
        s2_input_gt = crop_resized_gt.unsqueeze(0).to(device)

        gt_tensor_iso = torch.tensor(gt_imp_crop_raw, dtype=torch.float32).unsqueeze(0)
        gt_imp_resized_iso = resize_transform(gt_tensor_iso)[0].numpy().astype(np.uint8)

        with torch.no_grad():
            s2_out_gt = stage2_model(s2_input_gt)
            stage2_pred_gt = torch.argmax(s2_out_gt, dim=1).detach().cpu().numpy().squeeze(0).astype(np.uint8)

        s2_metrics_isolated = calculate_metrics(gt_imp_resized_iso, stage2_pred_gt)

    print("-" * 65)
    print("Stage-2 Metrics (End-to-End: Crop from Predicted Stage-1 ROI):")
    print("-" * 65)
    print(f"Dice        : {s2_metrics_e2e['dice']:.4f}")
    print(f"IoU         : {s2_metrics_e2e['iou']:.4f}")
    print(f"Precision   : {s2_metrics_e2e['precision']:.4f}")
    print(f"Recall      : {s2_metrics_e2e['recall']:.4f}")
    print(f"Specificity : {s2_metrics_e2e['specificity']:.4f}")
    print(f"Accuracy    : {s2_metrics_e2e['accuracy']:.4f}")
    print(f"GT Volume   : {s2_metrics_e2e['gt_volume']} voxels")
    print(f"Pred Volume : {s2_metrics_e2e['pred_volume']} voxels")

    if s2_metrics_isolated:
        print("\n" + "-" * 65)
        print("Stage-2 Metrics (Isolated Benchmark: Crop from GT Implant — Cell 109):")
        print("-" * 65)
        print(f"Dice        : {s2_metrics_isolated['dice']:.4f}  (Notebook Validation: 0.7282 | Test: 0.7336)")
        print(f"IoU         : {s2_metrics_isolated['iou']:.4f}  (Notebook Validation: 0.5832 | Test: 0.5874)")
        print(f"Precision   : {s2_metrics_isolated['precision']:.4f}")
        print(f"Recall      : {s2_metrics_isolated['recall']:.4f}")
        print(f"Specificity : {s2_metrics_isolated['specificity']:.4f}")
        print(f"Accuracy    : {s2_metrics_isolated['accuracy']:.4f}")
        print(f"GT Volume   : {s2_metrics_isolated['gt_volume']} voxels")
        print(f"Pred Volume : {s2_metrics_isolated['pred_volume']} voxels")

    # -------------------------------------------------------------
    # BONE QUALITY CLASSIFICATION
    # -------------------------------------------------------------
    crop_resized_np = crop_resized_p1[0].detach().cpu().numpy()
    pred_implant_mask = (stage2_pred_p1 == 1)
    gt_implant_mask = (gt_imp_resized_p1 == 1)

    pred_mean = float(np.mean(crop_resized_np[pred_implant_mask])) if np.sum(pred_implant_mask) > 0 else 0.0
    gt_mean = float(np.mean(crop_resized_np[gt_implant_mask])) if np.sum(gt_implant_mask) > 0 else 0.0

    # Cell 105 classification (exact table in notebook screenshot)
    pred_class_105 = classify_bone_quality(pred_mean, mode="cell105")
    gt_class_105 = classify_bone_quality(gt_mean, mode="cell105")
    match_105 = (pred_class_105 == gt_class_105)

    # Cell 111 classification
    pred_class_111 = classify_bone_quality(pred_mean, mode="cell111")
    gt_class_111 = classify_bone_quality(gt_mean, mode="cell111")

    print("\n" + "-" * 65)
    print("Bone Quality Classification (Cell 105 Table Thresholds):")
    print("Thresholds: D1 > 0.6429 | D2 > 0.5286 | D3 > 0.3857 | D4 <= 0.3857")
    print("-" * 65)
    print(f"Pred Mean Intensity : {pred_mean:.6f} -> Class: {pred_class_105}")
    print(f"GT Mean Intensity   : {gt_mean:.6f} -> Class: {gt_class_105}")
    print(f"Classification Match: {'✅ MATCH' if match_105 else '❌ MISMATCH'}")

    # -------------------------------------------------------------
    # RUN PIPELINE (Full verification of return schema)
    # -------------------------------------------------------------
    print("\n[Pipeline Verification] Testing run_pipeline() schema...")
    pipe_res = run_pipeline(image_path, device=device)
    print("run_pipeline() result:")
    print(f"  - Bone Quality          : {pipe_res['bone_quality']}")
    print(f"  - Implant Volume Voxels : {pipe_res['implant_volume_voxels']}")
    print(f"  - ROI Mesh Vertices     : {len(pipe_res['roi_mesh']['vertices'])}, Faces: {len(pipe_res['roi_mesh']['faces'])}")
    print(f"  - Implant Mesh Vertices : {len(pipe_res['implant_mesh']['vertices'])}, Faces: {len(pipe_res['implant_mesh']['faces'])}")

    return {
        "case": case_idx,
        "image": image_path.name,
        "s1_metrics": s1_metrics,
        "s2_metrics": s2_metrics_e2e,
        "s2_metrics_isolated": s2_metrics_isolated,
        "pred_class": pred_class_105,
        "gt_class": gt_class_105,
        "class_match": match_105,
        "pipe_res": pipe_res,
    }


def main():
    parser = argparse.ArgumentParser(description="Verify two-stage ML pipeline against ground truth data.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=SCRIPT_DIR / "sample_data",
        help="Path to folder containing test .nrrd and .seg.nrrd files (default: ml-service/sample_data)",
    )
    parser.add_argument("--image", type=Path, default=None, help="Explicit path to a .nrrd image file")
    parser.add_argument("--label", type=Path, default=None, help="Explicit path to a .seg.nrrd ground truth file")
    parser.add_argument("--device", type=str, default=str(DEVICE), help="Computation device (cpu, cuda, mps)")

    args = parser.parse_args()
    device = torch.device(args.device)

    print("=" * 70)
    print("IMPLANT PLATFORM — ML PIPELINE VERIFICATION")
    print(f"Using Device : {device}")
    print("=" * 70)

    # 1. Check if explicit pair provided
    if args.image and args.label:
        if not args.image.exists():
            print(f"Error: Image file does not exist: {args.image}")
            sys.exit(1)
        if not args.label.exists():
            print(f"Error: Label file does not exist: {args.label}")
            sys.exit(1)
        pairs = [(args.image, args.label)]
    else:
        # Scan sample_data directory
        pairs = find_sample_pairs(args.data_dir)

    if not pairs:
        print(f"\nNo test pairs found in: {args.data_dir.resolve()}")
        print("\nTo run verification on real cases:")
        print("  1. Place test CBCT file(s) (*.nrrd) and segmentation file(s) (*.seg.nrrd)")
        print(f"     inside: {args.data_dir.resolve()}/")
        print("  2. Re-run: python ml-service/verify_pipeline.py")
        print("\nOr provide explicit paths:")
        print("  python ml-service/verify_pipeline.py --image /path/to/vol.nrrd --label /path/to/vol.seg.nrrd\n")
        return

    print(f"Found {len(pairs)} test case pair(s).\n")
    all_results = []
    for idx, (img_path, lbl_path) in enumerate(pairs, start=1):
        res = verify_single_case(img_path, lbl_path, case_idx=idx, device=device)
        all_results.append(res)

    # Summary table
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("OVERALL SUMMARY ACROSS TEST CASES")
        print("=" * 70)
        avg_s1_dice = np.mean([r["s1_metrics"]["dice"] for r in all_results])
        avg_s1_iou = np.mean([r["s1_metrics"]["iou"] for r in all_results])
        avg_s2_dice = np.mean([r["s2_metrics"]["dice"] for r in all_results])
        avg_s2_iou = np.mean([r["s2_metrics"]["iou"] for r in all_results])
        matches = sum(1 for r in all_results if r["class_match"])

        print(f"Mean Stage-1 ROI Dice     : {avg_s1_dice:.4f}  (Notebook Validation: 0.7775 | Test: 0.8491)")
        print(f"Mean Stage-1 ROI IoU      : {avg_s1_iou:.4f}  (Notebook Validation: 0.6933 | Test: 0.7715)")
        print(f"Mean Stage-2 Implant Dice : {avg_s2_dice:.4f}  (Notebook Validation: 0.7282 | Test: 0.7336)")
        print(f"Mean Stage-2 Implant IoU  : {avg_s2_iou:.4f}  (Notebook Validation: 0.5832 | Test: 0.5874)")
        print(f"Bone Quality Match Rate   : {matches}/{len(all_results)} ({matches/len(all_results)*100:.1f}%)")


if __name__ == "__main__":
    main()
