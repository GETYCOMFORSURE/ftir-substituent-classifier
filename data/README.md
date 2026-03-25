# Personally Collected Spectra — Unknown Compounds

These are IR spectra collected personally in the lab on the Shimadzu FTIR spectrometer.

## Files

- `A_1.txt` through `A_5.txt` — 5 repeat measurements of an unknown aminobenzoic acid isomer

## Instrument parameters

| Parameter | Value |
|---|---|
| Measurement Mode | % Transmittance |
| Apodization | Happ-Genzel |
| No. of Scans | 10 |
| Resolution | 0.9 cm⁻¹ |
| Range | 400–4000 cm⁻¹ |

## How these are used

In notebook `Machine_Learning_Part_II.ipynb`, the full training library (380 spectra) is used to train a k-NN classifier. These 5 spectra are then passed in as the test set, and the model predicts whether the unknown compound is ortho-, meta-, or para-aminobenzoic acid.
