#!/usr/bin/env python3
"""
Merlin Analysis Script for 430 Lung Segmentation Dataset

This script runs the Merlin model on a custom dataset of CT scans and generates
phenotype predictions with drop rate analysis.

Usage:
    python run_merlin_analysis.py <dataset_path> [--output_dir OUTPUT_DIR]
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from collections import defaultdict
import nibabel as nib

# Add merlin to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from merlin.models.load import Merlin


def setup_dataset_paths(dataset_path):
    """
    Discover all case folders in the dataset.
    
    Args:
        dataset_path (str): Path to dataset root directory
        
    Returns:
        list: Sorted list of case folder paths
    """
    dataset_path = Path(dataset_path)
    case_folders = sorted([p for p in dataset_path.iterdir() if p.is_dir() and p.name.startswith('Case_')])
    return case_folders


def load_model(device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Load the pre-trained Merlin model.
    
    Args:
        device (str): Device to load model on ('cuda' or 'cpu')
        
    Returns:
        tuple: (model, device)
    """
    print(f"Loading Merlin model on device: {device}")
    model = Merlin(PhenotypeCls=True)
    
    # Move model to device
    if device == 'cuda':
        model = model.cuda()
    else:
        model = model.cpu()
    
    model.eval()
    return model, device


def process_case(case_folder, model, cache_dir=None, device='cuda'):
    """
    Process a single case folder with Merlin model.
    
    Args:
        case_folder (Path): Path to case folder
        model: Merlin model instance
        cache_dir (Path): Optional cache directory for embeddings
        device (str): Device to use
        
    Returns:
        dict: Case results containing phenotypes and scores
    """
    import nibabel as nib
    
    case_name = case_folder.name
    
    # Look for CT images in case folder
    image_files = list(case_folder.glob('*.nii')) + list(case_folder.glob('*.nii.gz'))
    
    if not image_files:
        return None
    
    try:
        # Load image
        image_path = image_files[0]
        img = nib.load(str(image_path))
        img_data = torch.from_numpy(img.get_fdata()).float()
        
        # Ensure 3D
        if img_data.dim() == 4:
            img_data = img_data[:, :, :, 0]
        
        # Downsample large volumes to prevent OOM
        if img_data.shape[2] > 400:
            indices = torch.linspace(0, img_data.shape[2]-1, 400, dtype=torch.long)
            img_data = img_data[:, :, indices]
        
        # Normalize to [-1, 1]
        img_data = torch.clamp(img_data, -1000, 400)
        img_data = (img_data + 1000) / 1400
        
        # Add batch and channel dimensions
        img_data = img_data.unsqueeze(0).unsqueeze(0)
        img_data = img_data.to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model.model(img_data)
        
        # Clear GPU cache after processing
        torch.cuda.empty_cache()
        
        # Extract phenotype predictions
        if isinstance(outputs, torch.Tensor):
            scores = outputs.cpu().squeeze().numpy()
            
            # Get top 10 phenotypes
            top_indices = np.argsort(-scores)[:10]
            
            phenotypes = []
            for rank, idx in enumerate(top_indices, 1):
                phenotypes.append({
                    'rank': rank,
                    'phencode': f'{idx}',
                    'score': float(scores[idx]),
                    'description': f'Phenotype {idx}'
                })
        else:
            phenotypes = []
        
        return {
            'case_name': case_name,
            'phenotypes': phenotypes,
            'success': True
        }
    except Exception as e:
        # Clear GPU cache on error
        torch.cuda.empty_cache()
        return {
            'case_name': case_name,
            'error': str(e),
            'success': False
        }


def analyze_predictions(results):
    """
    Analyze model predictions and extract top phenotypes with drop rates.
    
    Args:
        results (list): List of case results
        
    Returns:
        tuple: (phenotypes_df, dropoff_df)
    """
    phenotypes_data = []
    dropoff_data = []
    
    for result in results:
        if not result or not result['success']:
            continue
            
        case_name = result['case_name']
        phenotypes = result.get('phenotypes', [])
        
        # Store all phenotypes for this case
        for pheno in phenotypes:
            phenotypes_data.append({
                'Case': case_name,
                'Rank': pheno['rank'],
                'Phencode': pheno['phencode'],
                'Description': pheno['description'],
                'Score': pheno['score']
            })
        
        # Calculate drop rates from top 10
        if len(phenotypes) > 3:
            top_3_scores = [p['score'] for p in phenotypes[:3]]
            rest_scores = [p['score'] for p in phenotypes[3:]]
            
            top_3_avg = np.mean(top_3_scores) if top_3_scores else 0
            rest_avg = np.mean(rest_scores) if rest_scores else 0
            
            if top_3_avg > 0:
                dropoff_percent = ((top_3_avg - rest_avg) / top_3_avg) * 100
            else:
                dropoff_percent = 0
                
            dropoff_data.append({
                'Case': case_name,
                'Top_3_Avg': top_3_avg,
                'Rest_Avg': rest_avg,
                'Dropoff_Percent': dropoff_percent
            })
    
    phenotypes_df = pd.DataFrame(phenotypes_data)
    dropoff_df = pd.DataFrame(dropoff_data)
    
    return phenotypes_df, dropoff_df


def extract_global_top_phenotypes(phenotypes_df, top_n=10):
    """
    Extract the top N phenotypes across all cases.
    
    Args:
        phenotypes_df (DataFrame): Phenotypes data for all cases
        top_n (int): Number of top phenotypes to extract
        
    Returns:
        DataFrame: Top phenotypes with frequency and average scores
    """
    pheno_stats = phenotypes_df.groupby('Phencode').agg({
        'Description': 'first',
        'Score': ['mean', 'std', 'count'],
        'Rank': 'mean'
    }).reset_index()
    
    pheno_stats.columns = ['Phencode', 'Description', 'Avg_Score', 'Std_Score', 'Frequency', 'Avg_Rank']
    pheno_stats = pheno_stats.sort_values('Avg_Score', ascending=False).head(top_n)
    
    return pheno_stats


def extract_top_phenotypes_by_dropoff(dropoff_df, phenotypes_df, top_n=10):
    """
    Extract top phenotypes ranked by average drop-off rate.
    
    Args:
        dropoff_df (DataFrame): Drop-off rates data
        phenotypes_df (DataFrame): Phenotypes data
        top_n (int): Number to extract
        
    Returns:
        DataFrame: Top phenotypes with drop-off metrics
    """
    # Get top phenotypes from cases with highest drop-off rates
    top_cases = dropoff_df.nlargest(int(len(dropoff_df) * 0.3), 'Dropoff_Percent')['Case'].tolist()
    top_phenos = phenotypes_df[phenotypes_df['Case'].isin(top_cases)]
    
    pheno_summary = top_phenos.groupby('Phencode').agg({
        'Description': 'first',
        'Score': 'mean',
        'Rank': 'mean'
    }).reset_index()
    
    pheno_summary = pheno_summary.sort_values('Rank').head(top_n)
    
    return pheno_summary


def main():
    parser = argparse.ArgumentParser(description='Run Merlin analysis on custom CT dataset')
    parser.add_argument('dataset_path', help='Path to dataset root directory')
    parser.add_argument('--output_dir', default=None, help='Output directory for results')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for inference')
    parser.add_argument('--use_cache', action='store_true', help='Use cached embeddings if available')
    
    args = parser.parse_args()
    
    # Setup paths
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir) if args.output_dir else dataset_path
    cache_dir = dataset_path / 'cache' if args.use_cache else None
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("MERLIN ANALYSIS - Custom CT Dataset")
    print(f"{'='*60}")
    print(f"Dataset Path: {dataset_path}")
    print(f"Output Directory: {output_dir}")
    print(f"Device: {args.device}")
    print(f"Use Cache: {args.use_cache}")
    print(f"{'='*60}\n")
    
    # Load model
    model, device = load_model(device=args.device)
    
    # Discover cases
    case_folders = setup_dataset_paths(dataset_path)
    print(f"Found {len(case_folders)} cases to process\n")
    
    # Process cases
    results = []
    print("Processing cases...")
    for case_folder in tqdm(case_folders, desc="Cases"):
        result = process_case(case_folder, model, cache_dir, device=device)
        if result:
            results.append(result)
    
    print(f"\nSuccessfully processed {sum(1 for r in results if r.get('success', False))} cases\n")
    
    # Analyze predictions
    print("Analyzing predictions...")
    phenotypes_df, dropoff_df = analyze_predictions(results)
    
    # Extract top phenotypes
    print("Extracting top phenotypes...")
    top_phenos_by_score = extract_global_top_phenotypes(phenotypes_df, top_n=10)
    top_phenos_by_dropoff = extract_top_phenotypes_by_dropoff(dropoff_df, phenotypes_df, top_n=10)
    
    # Save results
    print("Saving results...")
    import os, time
    # Try to save with retries in case file is locked
    temp_suffix = f"_{int(time.time())}"
    phenotypes_df.to_csv(output_dir / f'phenotype_predictions{temp_suffix}.csv', index=False)
    dropoff_df.to_csv(output_dir / f'dropoff_rates{temp_suffix}.csv', index=False)
    top_phenos_by_score.to_csv(output_dir / f'top_phenotypes_by_score{temp_suffix}.csv', index=False)
    top_phenos_by_dropoff.to_csv(output_dir / f'top_phenotypes_by_dropoff{temp_suffix}.csv', index=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print("TOP 10 PHENOTYPES BY AVERAGE SCORE")
    print(f"{'='*60}")
    print(top_phenos_by_score.to_string(index=False))
    
    print(f"\n{'='*60}")
    print("TOP 10 PHENOTYPES BY DROP-OFF RATE")
    print(f"{'='*60}")
    print(top_phenos_by_dropoff.to_string(index=False))
    
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"Total Cases: {len(case_folders)}")
    print(f"Successfully Processed: {len(phenotypes_df.groupby('Case'))}")
    print(f"Total Phenotypes Found: {phenotypes_df['Phencode'].nunique()}")
    print(f"Average Drop-off Rate: {dropoff_df['Dropoff_Percent'].mean():.2f}%")
    print(f"Median Drop-off Rate: {dropoff_df['Dropoff_Percent'].median():.2f}%")
    print(f"Min Drop-off Rate: {dropoff_df['Dropoff_Percent'].min():.2f}%")
    print(f"Max Drop-off Rate: {dropoff_df['Dropoff_Percent'].max():.2f}%")
    print(f"{'='*60}\n")
    
    print(f"Results saved to: {output_dir}")
    print("Files created:")
    print(f"  - phenotype_predictions.csv")
    print(f"  - dropoff_rates.csv")
    print(f"  - top_phenotypes_by_score.csv")
    print(f"  - top_phenotypes_by_dropoff.csv")


if __name__ == '__main__':
    main()
