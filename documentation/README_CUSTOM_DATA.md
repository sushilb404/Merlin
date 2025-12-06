# Running Merlin on Custom Data

This guide covers how to run Merlin inference on your own medical imaging datasets.

## Overview

Merlin can generate radiology reports for custom CT scans. This document explains:
- How to prepare your data
- How to run inference
- How to interpret results

## Data Format Requirements

### Directory Structure

```
your_data/
├── Case_001/
│   └── image.nii.gz              # 3D CT volume
├── Case_002/
│   └── image.nii.gz
└── Case_003/
    └── image.nii.gz
```

### Image Format

- **Format**: NIfTI (.nii, .nii.gz) or DICOM (.dcm)
- **Dimensions**: 3D volumes (e.g., 512×512×200 slices)
- **Data Type**: uint16 or float32 (HU units for CT)
- **Spacing**: 1-2mm recommended (automatic resampling available)

### Preprocessing

Merlin expects:
- **HU Range**: Clipped to [-1000, 400] (lung window)
- **Orientation**: Standard radiological orientation
- **Single Phase**: For multi-phase, average or select arterial phase

## Quick Start

### 1. Prepare Your Data

```bash
# Example: Convert DICOM to NIfTI
dcm2niix -o ./data/Case_001 ./dicom_source/patient_001/
```

### 2. Run Merlin Analysis

```python
from merlin.models import load_model
import torch

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_model(device=device)

# Run inference
case_path = './data/Case_001'
predictions = model.predict(case_path)

# Print top 5 phenotypes
for i, pheno in enumerate(predictions['phenotypes'][:5], 1):
    print(f"{i}. {pheno['description']} (score: {pheno['score']:.3f})")
```

### 3. Batch Processing

```bash
python documentation/run_merlin_analysis.py ./your_data --device cuda
```

## Output Format

### phenotype_predictions.csv

```
Case,Rank,Phencode,Description,Score
Case_001,1,786.05,Dyspnea and respiratory abnormalities,0.897
Case_001,2,518.3,Pneumonia unspecified,0.754
Case_001,3,786.09,Other symptoms involving respiratory system,0.683
```

### dropoff_rates.csv

```
Case,Top_3_Avg,Rest_Avg,Dropoff_Percent
Case_001,0.778,0.042,94.6
Case_002,0.612,0.031,94.9
```

**Dropoff Interpretation**:
- **>95%**: Very confident predictions
- **85-95%**: Confident predictions
- **<85%**: Multiple plausible phenotypes

## Performance Tips

### Speed Optimization
- Use CUDA GPU if available (10-20x faster than CPU)
- Enable caching: `model.predict(path, cache=True)`
- Process in batches for large datasets

### Memory Optimization
- Large volumes (>1000 slices): Downsample to 512×512×400
- Use float32 instead of float64
- Process cases sequentially if needed

### Quality Optimization
- Ensure proper HU range normalization
- Use consistent image spacing
- Remove severely corrupted images
- Verify segmentation masks (if available)

## Common Issues

### Issue: "Image not found"
**Solution**: Check file naming and format
```bash
# List files in case folder
ls Case_001/
# Should show: image.nii.gz or similar
```

### Issue: "OOM" (Out of Memory)
**Solution**: Reduce image size
```python
# Resample to smaller size
# Edit the preprocessing in your pipeline
```

### Issue: Unexpected results
**Solution**: Verify preprocessing
- Check HU clipping
- Verify image orientation
- Compare with expected ranges

## Advanced Usage

### Custom Models

To use a fine-tuned model:

```python
from merlin.models import load_model

# Load custom checkpoint
model = load_model(checkpoint_path='path/to/checkpoint.pt')
predictions = model.predict(case_path)
```

### Batch API

```python
from merlin.models import load_model
import pandas as pd

model = load_model()

# Process multiple cases
results = []
for case_folder in case_folders:
    pred = model.predict(case_folder)
    results.append({
        'case': case_folder.name,
        'top_phenotype': pred['phenotypes'][0]['description'],
        'score': pred['phenotypes'][0]['score']
    })

df = pd.DataFrame(results)
df.to_csv('results.csv', index=False)
```

### Export to Different Formats

```python
import json
import pandas as pd

# Load predictions
predictions = model.predict(case_path)

# Export to JSON
with open('predictions.json', 'w') as f:
    json.dump(predictions, f, indent=2)

# Export to CSV
df = pd.DataFrame([
    {
        'phencode': p['phencode'],
        'description': p['description'],
        'score': p['score']
    } for p in predictions['phenotypes']
])
df.to_csv('predictions.csv', index=False)
```

## Your Dataset: NTM_lungSeg (430 Cases)

### Dataset Overview
- **Cases**: 430 lung CT scans
- **Format**: NIfTI files with segmentation masks
- **Cache**: Pre-computed embeddings available in `cache/`
- **Previous Results**: See `phenotype_predictions.csv` and `dropoff_rates.csv`

### Processing Your Dataset

```bash
# Process all 430 cases
python documentation/run_merlin_analysis.py \
  "C:\Users\sbohar3\Downloads\BMI555_DATA_SUSHIL\BMI555_DATA_SUSHIL\NTM_lungSeg" \
  --device cuda \
  --use_cache

# Expected runtime: 7-15 hours on GPU
```

### Analyzing Results

After processing:

```python
import pandas as pd

# Load results
phenotypes = pd.read_csv('phenotype_predictions.csv')
dropoff = pd.read_csv('dropoff_rates.csv')

# Top phenotypes across all 430 cases
top_phenos = phenotypes.groupby('Phencode').size().sort_values(ascending=False).head(10)

# Cases with highest uncertainty
uncertain = dropoff.nsmallest(10, 'Dropoff_Percent')

# Cases with highest certainty
certain = dropoff.nlargest(10, 'Dropoff_Percent')
```

## File Organization

After running analysis on your dataset:

```
NTM_lungSeg/
├── cache/                          # Embeddings cache
│   ├── [hash1].pt
│   ├── [hash2].pt
│   └── ... (430 files)
├── Case_001/ to Case_430/          # Case folders
├── phenotype_predictions.csv       # All predictions (top 10 per case)
├── dropoff_rates.csv              # Drop-off metrics
├── top_phenotypes_by_score.csv    # Global top 10 by score
└── top_phenotypes_by_dropoff.csv  # Top 10 from high-confidence cases
```

## Reproducibility

To ensure reproducible results:

```python
import torch
import numpy as np

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

# Run prediction
predictions = model.predict(case_path)
```

## References

### Related Files
- `00_START_HERE.md` - Overall Merlin guide
- `SETUP_GUIDE.md` - Installation and setup
- `run_merlin_analysis.py` - Main analysis script

### Literature
- Merlin paper: [Citation]
- ICD-9 phenotype codes: [Reference]
- CT preprocessing: [Reference]

## Support

For detailed guides:
1. See `SETUP_GUIDE.md` for setup instructions
2. See `00_START_HERE.md` for general information
3. Check `inference.md` for inference details
4. Review the main repository: [GitHub](https://github.com/sushilb404/Merlin)

---

**Last Updated**: December 5, 2025
**For**: 430-case NTM_lungSeg dataset
