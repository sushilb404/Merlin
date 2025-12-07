#!/usr/bin/env python3
r"""
Extract Merlin image embeddings for NTM and TB datasets and save to CSV.

Default base_dir:
- C:\Users\sbohar3\Downloads\BMI_555_Sushil

Expected subfolders under base_dir (auto-detected if present):
- NTM_lungSeg
- TB_lungSeg_part1
- TB_lungSeg_part2

Usage (PowerShell, in venv):
  python documentation/extract_embeddings.py

Custom datasets:
  python documentation/extract_embeddings.py --dataset NTM "C:\\path\\to\\ntm" --dataset TB "C:\\path\\to\\tb1" --dataset TB "C:\\path\\to\\tb2"
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import nibabel as nib
import pandas as pd
import torch
from tqdm import tqdm

# Add repository root to path so we can import merlin
sys.path.insert(0, str(Path(__file__).parent.parent))
from merlin.models.load import Merlin  # noqa: E402


def find_cases(root: Path) -> List[Path]:
    """Return immediate subfolders that contain at least one NIfTI or DICOM file."""
    cases: List[Path] = []
    if not root.exists():
        return cases
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if list(p.glob("*.nii")) or list(p.glob("*.nii.gz")) or list(p.glob("*.dcm")):
            cases.append(p)
    return sorted(cases)


def load_model(device: str):
    model = Merlin(ImageEmbedding=True)
    model = model.to(device)
    model.eval()
    return model


def load_volume(case_dir: Path) -> torch.Tensor:
    """Load a 3D volume tensor (1,1,D,H,W) normalized to [0,1]."""
    image_files = list(case_dir.glob("*.nii")) + list(case_dir.glob("*.nii.gz"))
    if not image_files:
        raise FileNotFoundError(f"No NIfTI found in {case_dir}")
    img = nib.load(str(image_files[0]))
    data = torch.from_numpy(img.get_fdata()).float()

    # If 4D, take first channel
    if data.dim() == 4:
        data = data[..., 0]

    # Downsample depth to <=400 slices to avoid OOM
    if data.shape[2] > 400:
        idx = torch.linspace(0, data.shape[2] - 1, 400, dtype=torch.long)
        data = data[:, :, idx]

    # Clamp HU and normalize to [0,1]
    data = torch.clamp(data, -1000, 400)
    data = (data + 1000) / 1400

    # Add batch and channel dims -> (1,1,D,H,W)
    data = data.unsqueeze(0).unsqueeze(0)
    return data


@torch.inference_mode()
def extract_embedding(model, volume: torch.Tensor, device: str) -> List[float]:
    volume = volume.to(device)
    outputs = model(volume)
    if isinstance(outputs, torch.Tensor):
        feats = outputs.flatten().detach().cpu()
    elif isinstance(outputs, (list, tuple)) and outputs:
        feats = torch.as_tensor(outputs[0]).flatten().detach().cpu()
    else:
        raise RuntimeError("Model did not return a tensor")
    return feats.tolist()


def parse_args():
    parser = argparse.ArgumentParser(description="Extract Merlin embeddings for NTM and TB")
    parser.add_argument(
        "--dataset",
        action="append",
        nargs=2,
        metavar=("GROUP", "PATH"),
        help="Add dataset with group label, e.g., --dataset NTM C:\\path\\ntm",
    )
    parser.add_argument(
        "--base_dir",
        type=Path,
        default=Path(r"C:\Users\sbohar3\Downloads\BMI_555_Sushil"),
        help="Base directory containing NTM_lungSeg, TB_lungSeg_part1, TB_lungSeg_part2",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path(r"C:\Users\sbohar3\merlin\results\embeddings\ntm_tb_embeddings.csv"),
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing CSV instead of overwriting",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="Device to use (cuda or cpu). Defaults to cuda if available, else cpu",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    def build_default_datasets(base_dir: Path) -> List[Tuple[str, Path]]:
        candidates = [
            ("NTM", base_dir / "NTM_lungSeg"),
            ("TB", base_dir / "TB_lungSeg_part1"),
            ("TB", base_dir / "TB_lungSeg_part2"),
        ]
        return [(grp, p) for grp, p in candidates if p.exists()]

    default_datasets: List[Tuple[str, Path]] = build_default_datasets(args.base_dir)
    if args.dataset:
        datasets = [(grp, Path(p)) for grp, p in args.dataset]
    else:
        datasets = default_datasets

    if not datasets:
        raise ValueError("No datasets found. Provide --dataset entries or ensure expected subfolders exist under base_dir.")

    model = load_model(device)
    rows = []

    for group, root in datasets:
        root = Path(root)
        cases = find_cases(root)
        print(f"{group}: found {len(cases)} cases under {root}")
        for case_dir in tqdm(cases, desc=f"{group} cases"):
            try:
                vol = load_volume(case_dir)
                emb = extract_embedding(model, vol, device)
                rows.append({
                    "case_id": case_dir.name,
                    "group": group,
                    "embedding": json.dumps(emb),
                })
            except Exception as e:
                print(f"Failed {case_dir}: {e}")
            finally:
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    if args.append and args.output_csv.exists():
        existing_df = pd.read_csv(args.output_csv)
        df = pd.concat([existing_df, df], ignore_index=True)
        print(f"Appended {len(rows)} new embeddings to {args.output_csv}")
    
    df.to_csv(args.output_csv, index=False)
    print(f"Saved embeddings to {args.output_csv}")


if __name__ == "__main__":
    main()
