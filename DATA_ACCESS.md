# TB vs NTM Classifier - Data Access

## Large Files

The embeddings CSV file is too large for GitHub (52.6 MB). It's stored locally in:
```
results/embeddings/ntm_tb_embeddings.csv
```

### Access Options

**For Local Development:**
- Run the extraction scripts yourself (see `documentation/extract_embeddings.py`)
- NTM extraction: ~2.5 hours on CPU
- TB extraction: ~4.5 hours on CPU

**For Remote Access:**

Option 1: **Hugging Face Datasets** (Recommended for ML)
```bash
# 1. Create HF account: https://huggingface.co
# 2. Get token: https://huggingface.co/settings/tokens
# 3. Upload:
$env:HF_TOKEN = "your_token"
python documentation/upload_to_huggingface.py
```

Option 2: **Google Drive**
- Upload to Google Drive
- Share link in this README

Option 3: **Zenodo** (Academic + DOI)
- Upload dataset to zenodo.org
- Get citable DOI

## Dataset Contents

**ntm_tb_embeddings.csv** (52.6 MB)
- 1,307 total embeddings
- 430 NTM lung cases
- 877 TB lung cases
- Features: group, case_path, embedding (2048-dim Merlin features)

## Classification Results

**Performance:**
- AUC Score: 0.784 (good separation)
- Train/Test Split: 80/20
- Model: Logistic Regression on standardized embeddings

**Visualizations:**
- `tb_vs_ntm_roc.png` - ROC curve
- `tb_vs_ntm_tsne.png` - t-SNE 2D projection
- `tb_vs_ntm_umap.png` - UMAP projection
- `tb_vs_ntm_metrics.txt` - Detailed metrics

## How to Reproduce

```bash
# 1. Extract embeddings
python documentation/extract_embeddings.py --device cpu

# 2. Train classifier
python documentation/train_tb_vs_ntm.py

# 3. View results in results/embeddings/
```

## Contact

For questions or to request data access, contact the repository maintainer.
