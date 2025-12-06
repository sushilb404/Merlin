# Merlin - Radiology Report Generation with Custom Data

Welcome to Merlin! This guide will help you get started with running Merlin on your custom CT dataset.

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Dataset Preparation](#dataset-preparation)
3. [Running Analysis](#running-analysis)
4. [Understanding Results](#understanding-results)
5. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (for GPU acceleration) or CPU
- 16GB+ RAM recommended
- Storage for model checkpoints (~2GB)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sushilb404/Merlin.git
   cd Merlin
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1  # Windows
   # or
   source venv/bin/activate     # Linux/Mac
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip setuptools
   pip install -e .
   ```

4. **Download pre-trained model** (optional, auto-downloads on first run):
   ```bash
   python -c "from merlin.models import load_model; load_model()"
   ```

---

## 📁 Dataset Preparation

### Dataset Structure

Your dataset should be organized as follows:

```
your_dataset/
├── cache/                    # Cached embeddings (created automatically)
├── Case_001/
│   ├── image.nii.gz         # CT scan (NIfTI format)
│   └── metadata.json        # (Optional) Case metadata
├── Case_002/
│   ├── image.nii.gz
│   └── metadata.json
├── ...
└── Case_N/
    ├── image.nii.gz
    └── metadata.json
```

### Supported Formats

- **Medical Images**: NIfTI (.nii, .nii.gz), DICOM (.dcm)
- **Image Specifications**:
  - 3D CT volumes (axial acquisition preferred)
  - Typical resolution: 512×512×100-400 slices
  - Single phase or multi-phase (averaged)

### Data Preprocessing Tips

- **Normalization**: HU clipping to [-1000, 400] is recommended
- **Resampling**: 1-2mm slice spacing preferred
- **Memory**: Large volumes (>1000 slices) may need downsampling
- **Quality**: Remove severely motion-corrupted scans

---

## 🔍 Running Analysis

### Basic Usage

```bash
python documentation/run_merlin_analysis.py <dataset_path> [options]
```

### Example

```bash
python documentation/run_merlin_analysis.py "C:\Users\sbohar3\Downloads\BMI555_DATA_SUSHIL\NTM_lungSeg" --device cuda --use_cache
```

### Command Options

| Option | Description | Default |
|--------|-------------|---------|
| `dataset_path` | Path to dataset root directory | Required |
| `--output_dir` | Directory for output files | Same as dataset |
| `--device` | `cuda` or `cpu` | Auto-detect |
| `--use_cache` | Use cached embeddings if available | False |

### Running on Your 430 Dataset

```bash
cd C:\Users\sbohar3\merlin
.\venv\Scripts\Activate.ps1

# Run Merlin on your 430 cases
python documentation/run_merlin_analysis.py "C:\Users\sbohar3\Downloads\BMI555_DATA_SUSHIL\BMI555_DATA_SUSHIL\NTM_lungSeg" --use_cache
```

**Expected Runtime**: 
- GPU (CUDA): ~1-3 minutes per case (~7-15 hours total for 430 cases)
- CPU: ~5-10 minutes per case (~35-70 hours total)

---

## 📊 Understanding Results

### Output Files

After running the analysis, you'll find these files in your dataset directory:

#### 1. **phenotype_predictions.csv**
Top 10 phenotypes for each case.

| Column | Description |
|--------|-------------|
| Case | Case identifier (e.g., Case_001) |
| Rank | Phenotype rank (1-10) |
| Phencode | ICD-9 phenotype code |
| Description | Human-readable phenotype description |
| Score | Model confidence score (0-1) |

**Example**:
```
Case,Rank,Phencode,Description,Score
Case_001,1,786.05,Dyspnea,0.8942
Case_001,2,518.3,Pneumonia unspecified,0.7235
Case_001,3,786.09,Other respiratory symptoms,0.6512
```

#### 2. **dropoff_rates.csv**
Confidence drop-off analysis between top 3 and remaining phenotypes.

| Column | Description |
|--------|-------------|
| Case | Case identifier |
| Top_3_Avg | Average score of top 3 phenotypes |
| Rest_Avg | Average score of remaining phenotypes |
| Dropoff_Percent | Drop-off percentage = ((Top_3_Avg - Rest_Avg) / Top_3_Avg) × 100 |

**Interpretation**:
- **High drop-off (>95%)**: Model is very confident in top 3 phenotypes
- **Medium drop-off (85-95%)**: Moderate confidence, more options possible
- **Low drop-off (<85%)**: Many phenotypes are plausible

**Example**:
```
Case,Top_3_Avg,Rest_Avg,Dropoff_Percent
Case_001,0.7563,0.0198,97.38
Case_002,0.5421,0.0156,97.13
```

#### 3. **top_phenotypes_by_score.csv**
Global top 10 phenotypes across all 430 cases (ranked by average score).

| Column | Description |
|--------|-------------|
| Phencode | ICD-9 code |
| Description | Phenotype description |
| Avg_Score | Average confidence score across all cases |
| Std_Score | Standard deviation of scores |
| Frequency | Number of cases where this phenotype appeared |
| Avg_Rank | Average ranking in top 10 |

**Use Case**: Identifies the most prevalent phenotypes in your dataset.

#### 4. **top_phenotypes_by_dropoff.csv**
Top 10 phenotypes from cases with highest drop-off rates.

**Use Case**: Identifies phenotypes from "high-confidence" predictions.

---

## 📈 Key Metrics Explained

### Phenotype Score
- **Range**: 0.0 to 1.0
- **Interpretation**: 
  - > 0.8: Strong indicator (likely present)
  - 0.5-0.8: Moderate evidence
  - < 0.5: Weak evidence

### Drop-off Rate
- **Definition**: Percentage confidence reduction from top 3 to remaining phenotypes
- **High drop-off**: Model chose top 3 with high confidence
- **Low drop-off**: Multiple phenotypes have similar scores

### Case Examples

**Example 1 - High Confidence Case**:
```
Case_001: Drop-off = 97.4%
Top 3 phenotypes:
  1. Dyspnea (0.89)
  2. Pneumonia (0.72)
  3. Respiratory symptoms (0.65)
Average of Top 3: 0.75
Average of Rest: 0.02
→ Model is very certain about these 3 phenotypes
```

**Example 2 - Uncertain Case**:
```
Case_050: Drop-off = 72.3%
Top 3 phenotypes:
  1. Abnormal findings (0.45)
  2. Other findings (0.42)
  3. Additional findings (0.40)
Average of Top 3: 0.42
Average of Rest: 0.12
→ Multiple phenotypes are plausible
```

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. "Model not found" Error
**Solution**: Download model checkpoint manually
```bash
python -c "from merlin.models import load_model; load_model()"
```

#### 2. Out of Memory (OOM) Error
**Solutions**:
- Reduce batch size (edit script)
- Use CPU instead: `--device cpu`
- Downsample large images
- Process in smaller batches

#### 3. No Images Found in Cases
**Check**:
- Images are in `.nii`, `.nii.gz`, or `.dcm` format
- Files are directly in case folders (not subdirectories)
- File permissions are correct

#### 4. Slow Processing
**Optimization**:
- Enable GPU: Ensure `--device cuda` is set
- Use cache: `--use_cache` flag
- Close other applications
- Consider batch processing

#### 5. Different Results Than Expected
**Factors**:
- Model is stochastic (slight variations are normal)
- Image preprocessing affects results
- Use seed setting for reproducibility

---

## 🔧 Advanced Usage

### Custom Processing Script

For more control, edit `documentation/run_merlin_analysis.py`:

```python
from merlin.models import load_model
import pandas as pd

# Load model
model = load_model(device='cuda')

# Custom processing
for case_folder in case_folders:
    predictions = model.predict(case_folder)
    # Process predictions...
```

### Batch Processing

Process multiple datasets:
```bash
for dataset in dataset1 dataset2 dataset3; do
    python documentation/run_merlin_analysis.py "$dataset"
done
```

### Export to Different Formats

Convert results to other formats:
```python
import pandas as pd

df = pd.read_csv('phenotype_predictions.csv')
df.to_excel('results.xlsx', index=False)  # Excel
df.to_json('results.json', orient='records')  # JSON
```

---

## 📚 Reference

### Model Details

- **Architecture**: Inflated 3D ResNet + Transformer
- **Pre-training**: Merlin Abdominal CT Dataset (25K scans)
- **Output**: ICD-9 phenotype predictions
- **Training Details**: See `SETUP_GUIDE.md`

### Dataset Details

Your dataset:
- **Name**: NTM_lungSeg
- **Cases**: 430 lung CT scans
- **Format**: NIfTI with segmentations
- **Previous Results**: Cached in `cache/` folder

### Citation

If using Merlin in your research:

```bibtex
@article{merlin2024,
  title={Merlin: Multimodal Radiology Report Learning},
  author={[Authors]},
  year={2024}
}
```

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review `SETUP_GUIDE.md` for detailed setup
3. Check GitHub issues: [sushilb404/Merlin](https://github.com/sushilb404/Merlin)
4. Review documentation in `documentation/` folder

---

## 📝 Next Steps

1. ✅ Prepare your dataset in the required format
2. ✅ Run `python documentation/run_merlin_analysis.py <dataset_path>`
3. ✅ Analyze results in the generated CSV files
4. ✅ Review top phenotypes and drop-off rates
5. ✅ Export results for further analysis

---

**Last Updated**: December 5, 2025
**Version**: 1.0
**Status**: Ready for 430-case dataset analysis
