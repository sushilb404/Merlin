#!/usr/bin/env python3
"""
Upload embeddings and results to Hugging Face Datasets.
Requires: huggingface_hub, and HF_TOKEN environment variable set.

Setup:
1. Create account at https://huggingface.co
2. Create a dataset repository (e.g., "merlin-tb-ntm-embeddings")
3. Get your user access token from https://huggingface.co/settings/tokens
4. Set environment variable: $env:HF_TOKEN = "your_token_here"

Usage:
  python upload_to_huggingface.py
"""

import os
from pathlib import Path

from huggingface_hub import HfApi, login


def upload_dataset():
    """Upload embeddings and results to Hugging Face."""
    # Get HF token from environment
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Error: Set HF_TOKEN environment variable first")
        print("  $env:HF_TOKEN = 'your_token_here'")
        print("Get token from: https://huggingface.co/settings/tokens")
        return

    # Login
    login(token=hf_token)

    # Files to upload
    repo_id = "sushilb404/merlin-tb-ntm-embeddings"  # Change to your username/repo
    files_to_upload = [
        ("results/embeddings/ntm_tb_embeddings.csv", "ntm_tb_embeddings.csv"),
        ("results/embeddings/tb_vs_ntm_metrics.txt", "tb_vs_ntm_metrics.txt"),
        ("results/embeddings/tb_vs_ntm_roc.png", "tb_vs_ntm_roc.png"),
        ("results/embeddings/tb_vs_ntm_tsne.png", "tb_vs_ntm_tsne.png"),
        ("results/embeddings/tb_vs_ntm_umap.png", "tb_vs_ntm_umap.png"),
    ]

    api = HfApi()

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        print(f"Using dataset: {repo_id}")
    except Exception as e:
        print(f"Error creating repo: {e}")
        return

    # Upload files
    for local_path, remote_name in files_to_upload:
        local_file = Path(local_path)
        if not local_file.exists():
            print(f"⚠️  Skipping {local_path} (not found)")
            continue

        try:
            api.upload_file(
                path_or_fileobj=str(local_file),
                path_in_repo=remote_name,
                repo_id=repo_id,
                repo_type="dataset",
            )
            print(f"✅ Uploaded {remote_name} ({local_file.stat().st_size / 1e6:.1f} MB)")
        except Exception as e:
            print(f"❌ Failed to upload {remote_name}: {e}")

    print(f"\n✨ Done! Access at: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    upload_dataset()
