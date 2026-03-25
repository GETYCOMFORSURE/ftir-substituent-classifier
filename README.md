# FTIR Substituent Classifier

Classifying benzene derivatives as *ortho-*, *meta-*, or *para-*substituted from their IR spectra — using PCA dimensionality reduction and k-Nearest Neighbours.

Built from scratch in Python as part of a third-year physical chemistry lab at the University of Oxford.

---

## What it does

Takes raw ATR-FTIR spectra (`.txt` files from a Shimadzu spectrometer) and runs them through a full ML pipeline:

1. **Pre-process** — interpolate to integer wavenumbers, normalise by area, crop to fingerprint region (630–880 cm⁻¹)
2. **Reduce** — PCA from 251 wavenumber features → 20 principal components
3. **Classify** — k-NN (k=3) trained on 76 compounds, 380 spectra total (5 repeats each)
4. **Predict** — classify personally collected spectra of unknown aminobenzoic acid isomers

PCA 3D scatter clearly separates the three substitution classes, confirming the fingerprint region encodes substituent position. k-NN achieves **75–90% accuracy** on held-out test sets.

---

## Structure

```
ftir-substituent-classifier/
├── notebook/
│   ├── Data_Pre_Processing_Functions.ipynb       # Interpolation + normalisation from scratch
│   ├── Data_Pre_Processing_Implementation.ipynb  # Bulk loading 380 spectra
│   ├── Data_Pre_Processing_Principal_Component_Analysis.ipynb  # PCA + 3D visualisation
│   ├── Machine_Learning_Part_I.ipynb             # Train/test split + k-NN
│   └── Machine_Learning_Part_II.ipynb            # Classifying personally collected unknowns
├── src/
│   └── ir_pipeline.py            # Core library: pre-processing, PCA, k-NN
├── data/
│   ├── raw_ir_spectra/           # 380 training spectra (not included — see below)
│   └── your_collected_spectra/   # Personally collected unknowns (A_1.txt – A_5.txt)
├── results/                      # PCA plots, model outputs
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/GETYCOMFORSURE/ftir-substituent-classifier.git
cd ftir-substituent-classifier
pip install -r requirements.txt
jupyter notebook notebook/
```

Or open any notebook directly in Google Colab. Mount your Drive and point `ir_pipeline.py` at your spectra folder.

---

## Data

Training spectra are not included in this repo (380 files, proprietary instrument exports). Place them in `data/raw_ir_spectra/` following the naming convention `compoundname_N.txt` (e.g. `p-xylene_1.txt`). Five personally collected unknown spectra are included in `data/your_collected_spectra/`.

---

*University of Oxford · Department of Chemistry*
