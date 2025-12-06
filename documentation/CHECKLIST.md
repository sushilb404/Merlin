# Merlin Project Checklist

Complete checklist for setting up and running Merlin on the 430 NTM_lungSeg dataset.

## ✅ Setup & Installation

- [ ] Clone Merlin repository
- [ ] Create virtual environment (venv or conda)
- [ ] Activate virtual environment
- [ ] Upgrade pip/setuptools: `pip install --upgrade pip setuptools`
- [ ] Install Merlin: `pip install -e .`
- [ ] Install dependencies: `pip install torch torchvision transformers monai nibabel pandas rich`
- [ ] Download pre-trained model: `python -c "from merlin.models import load_model; load_model()"`
- [ ] Verify GPU support (optional): `python -c "import torch; print(torch.cuda.is_available())"`

## 📁 Dataset Preparation

- [ ] Locate dataset: `C:\Users\sbohar3\Downloads\BMI555_DATA_SUSHIL\BMI555_DATA_SUSHIL\NTM_lungSeg`
- [ ] Verify 430 case folders exist (Case_001 to Case_430)
- [ ] Check cache folder exists with pre-computed embeddings
- [ ] Verify existing phenotype_predictions.csv and dropoff_rates.csv
- [ ] Inspect one case folder to understand structure
- [ ] Confirm all images are in NIfTI format (.nii or .nii.gz)

## 🚀 Running Analysis

### Initial Test Run
- [ ] Navigate to Merlin root directory
- [ ] Activate virtual environment
- [ ] Run analysis script (first 10 cases for testing):
  ```bash
  python documentation/run_merlin_analysis.py \
    "C:\Users\sbohar3\Downloads\BMI555_DATA_SUSHIL\BMI555_DATA_SUSHIL\NTM_lungSeg" \
    --device cuda \
    --use_cache
  ```
- [ ] Verify no errors in first 5 minutes
- [ ] Check that output files are being created

### Full Dataset Run
- [ ] Start full analysis on all 430 cases
- [ ] Monitor progress (expected: 7-15 hours on GPU)
- [ ] Check system resources periodically
- [ ] Verify no memory errors occur
- [ ] Allow script to complete fully

## 📊 Output Verification

After successful run, verify output files:

### Required Files
- [ ] `phenotype_predictions.csv` - Top 10 phenotypes per case
  - [ ] Contains all 430 cases
  - [ ] Has columns: Case, Rank, Phencode, Description, Score
  - [ ] All scores between 0-1
  
- [ ] `dropoff_rates.csv` - Drop-off metrics per case
  - [ ] Contains all 430 cases
  - [ ] Has columns: Case, Top_3_Avg, Rest_Avg, Dropoff_Percent
  - [ ] Drop-off rates between 0-100%

- [ ] `top_phenotypes_by_score.csv` - Global top 10
  - [ ] Contains top 10 phenotypes across all cases
  - [ ] Has columns: Phencode, Description, Avg_Score, Std_Score, Frequency, Avg_Rank

- [ ] `top_phenotypes_by_dropoff.csv` - Top 10 from high-confidence cases
  - [ ] Contains top 10 phenotypes from high drop-off rate cases

### Data Quality Checks
- [ ] No missing values in critical columns
- [ ] Phenotype descriptions are readable
- [ ] Scores are reasonable (mostly 0.3-0.9 range)
- [ ] Drop-off rates realistic (typically 85-98%)

## 📈 Analysis & Insights

### Statistical Analysis
- [ ] Calculate mean drop-off rate across all 430 cases
- [ ] Identify cases with highest certainty (top 10 by drop-off)
- [ ] Identify cases with lowest certainty (bottom 10 by drop-off)
- [ ] Count unique phenotypes found across dataset
- [ ] Identify most common phenotypes

### Top 10 Phenotypes Review
- [ ] List top 10 by score (most confident predictions)
- [ ] List top 10 by drop-off rate (from certain cases)
- [ ] Compare with domain knowledge (do they make sense for lung CT?)
- [ ] Note any unexpected phenotypes

### Case-Specific Insights
- [ ] Review a few high-certainty cases (drop-off > 95%)
- [ ] Review a few low-certainty cases (drop-off < 85%)
- [ ] Understand why model is certain/uncertain

## 📝 Documentation

### Existing Documentation Files
- [ ] `00_START_HERE.md` - Main getting started guide ✅ Created
- [ ] `SETUP_GUIDE.md` - Detailed setup instructions ✅ Created
- [ ] `README_CUSTOM_DATA.md` - Custom data guide ✅ Created
- [ ] `CHECKLIST.md` - This file ✅ Created

### Analysis Script
- [ ] `run_merlin_analysis.py` - Main analysis script ✅ Created
- [ ] Script handles all 430 cases
- [ ] Script generates all required outputs
- [ ] Script includes proper error handling
- [ ] Script includes progress bars

### Documentation Review
- [ ] 00_START_HERE.md includes quick start section
- [ ] 00_START_HERE.md includes troubleshooting
- [ ] SETUP_GUIDE.md includes system requirements
- [ ] SETUP_GUIDE.md includes GPU setup instructions
- [ ] README_CUSTOM_DATA.md includes data format specs
- [ ] README_CUSTOM_DATA.md includes performance tips

## 🧪 Testing & Validation

### Small Scale Testing
- [ ] Test script with 5 cases first
- [ ] Verify output format is correct
- [ ] Check for any runtime errors
- [ ] Measure performance (time per case)

### Full Scale Testing
- [ ] Run on all 430 cases
- [ ] Monitor memory usage
- [ ] Check for GPU OOM errors
- [ ] Verify completion without crashes

### Result Validation
- [ ] All cases have 10 phenotype predictions
- [ ] All cases have drop-off rates
- [ ] No NaN or invalid values
- [ ] Scores within expected range (0-1)
- [ ] Case names match input dataset

## 🔧 Troubleshooting

### If Issues Occur
- [ ] Check error messages in console
- [ ] Review troubleshooting section in documentation
- [ ] Check disk space (>50GB required)
- [ ] Monitor GPU memory usage
- [ ] Try CPU mode if GPU has issues
- [ ] Verify all dependencies installed

### If Model Loading Fails
- [ ] Manually download checkpoint
- [ ] Verify checkpoint file integrity
- [ ] Check cache directory permissions
- [ ] Try CPU mode first

### If Dataset Loading Fails
- [ ] Verify dataset path is correct
- [ ] Check case folder names match pattern
- [ ] Verify image files exist and are readable
- [ ] Check file permissions

## 📤 GitHub Upload

### Before Uploading
- [ ] All new files created locally
- [ ] All output CSVs generated successfully
- [ ] Analysis script runs without errors
- [ ] Documentation is complete and accurate

### Files to Commit
- [ ] `documentation/00_START_HERE.md` ✅
- [ ] `documentation/SETUP_GUIDE.md` ✅
- [ ] `documentation/README_CUSTOM_DATA.md` ✅
- [ ] `documentation/CHECKLIST.md` ✅
- [ ] `documentation/run_merlin_analysis.py` ✅
- [ ] `documentation/phenotype_predictions.csv` (after running)
- [ ] `documentation/dropoff_rates.csv` (after running)
- [ ] `documentation/top_phenotypes_by_score.csv` (after running)
- [ ] `documentation/top_phenotypes_by_dropoff.csv` (after running)

### Upload Steps
- [ ] Stage all new/modified files: `git add .`
- [ ] Commit with descriptive message: `git commit -m "Add Merlin analysis for 430 NTM_lungSeg dataset"`
- [ ] Push to GitHub: `git push origin main`
- [ ] Verify files appear on GitHub
- [ ] Check that large files (CSVs) uploaded correctly

## 📊 Final Deliverables

### Documentation (in documentation/ folder)
- [ ] `00_START_HERE.md` - Quick start guide for new users
- [ ] `SETUP_GUIDE.md` - Complete installation guide
- [ ] `README_CUSTOM_DATA.md` - Custom data processing guide
- [ ] `CHECKLIST.md` - This checklist
- [ ] `run_merlin_analysis.py` - Analysis script for any dataset

### Analysis Results (in dataset or documentation folder)
- [ ] `phenotype_predictions.csv` - All 430 cases with top 10 phenotypes
- [ ] `dropoff_rates.csv` - Confidence metrics for all 430 cases
- [ ] `top_phenotypes_by_score.csv` - Top 10 phenotypes globally
- [ ] `top_phenotypes_by_dropoff.csv` - Top 10 from confident cases

### Summary Report (optional)
- [ ] Key findings from 430 dataset analysis
- [ ] Most common phenotypes discovered
- [ ] Dataset characteristics (avg drop-off rate, etc.)
- [ ] Insights and recommendations

## 🎯 Success Criteria

✅ **Project Complete When**:
1. All documentation files created and accurate
2. Analysis script successfully runs on 430 cases
3. All output files generated without errors
4. All files uploaded to GitHub
5. Results can be accessed and reviewed
6. Documentation explains how to use results

---

## Notes

- **Dataset Path**: `C:\Users\sbohar3\Downloads\BMI555_DATA_SUSHIL\BMI555_DATA_SUSHIL\NTM_lungSeg`
- **Cases**: 430 lung CT scans (Case_001 to Case_430)
- **Expected Runtime**: 7-15 hours on GPU, 35-70 hours on CPU
- **Cache**: Available in `cache/` folder for speed
- **GPU Recommended**: Significantly faster than CPU

---

**Last Updated**: December 5, 2025
**Status**: Ready for execution
**Next Step**: Run full analysis on 430 dataset
