"""
C317.py — IR Spectroscopy Pre-Processing & PCA Library
=======================================================
Written as part of the C317 Physical Chemistry Computing Lab.

This library provides functions to:
  1. Interpolate raw IR spectra to integer wavenumber indices
  2. Normalise spectra by area (trapezium rule)
  3. Restrict to the fingerprint region (630-880 cm^-1)
  4. Bulk-load all spectra from a directory into a single DataFrame
  5. Apply PCA dimensionality reduction via sklearn

Usage:
    import sys
    sys.path.insert(0, 'src')
    from C317 import load_spectra, perform_pca

    # Load all pre-processed spectra (no PCA)
    df = load_spectra(n=0)

    # Load with PCA reduced to 20 components
    df_pca = load_spectra(n=20)
"""

import numpy as np
import pandas as pd
import os
import re
import sklearn.decomposition

# Path configuration
# Set this to the folder containing your raw .txt IR spectra files.
# When running locally: use "data/raw_ir_spectra"
# When running in Google Colab with Drive mounted: use your Drive path
SPECTRA_DIR = "data/raw_ir_spectra"


def interpolation(df_1):
    """
    Convert decimal wavenumber indices to integer wavenumbers via interpolation.

    The Shimadzu FTIR software exports wavenumbers to high precision
    (e.g. 400.232577 cm^-1). This function creates a new DataFrame with
    integer indices and linearly interpolates all transmittance values.

    Parameters
    ----------
    df_1 : pd.DataFrame
        Single-column DataFrame with decimal wavenumber index.

    Returns
    -------
    pd.DataFrame
        DataFrame with integer wavenumber index and interpolated values.
    """
    index_list = list(df_1.index)
    minimum_index = round(min(index_list))
    maximum_index = round(max(index_list))
    integer_indices = list(range(minimum_index, maximum_index + 1))

    nan_values = np.full(len(integer_indices), np.nan)
    df_2 = pd.DataFrame(
        data=nan_values,
        index=integer_indices,
        columns=[df_1.columns[0]]
    )

    df = pd.concat([df_1, df_2])
    df = df.sort_index()
    df = df.interpolate()
    df = df.loc[df.index.isin(integer_indices)]
    return df


def normalisation(df):
    """
    Normalise a spectrum so that the area under the curve equals 1.

    Uses the trapezium rule for numerical integration. The divisor
    (~327544.59) is the area of the reference spectrum computed in notebook 5a.

    Parameters
    ----------
    df : pd.DataFrame
        Pre-interpolated spectrum DataFrame.

    Returns
    -------
    pd.DataFrame
        Spectrum with transmittance values scaled to unit area.
    """
    def area_under_curve():
        x_values = df.index
        y_values = df.values.flatten()
        area = 0.0
        for i in range(1, len(x_values)):
            area += 0.5 * (y_values[i] + y_values[i - 1]) * (x_values[i] - x_values[i - 1])
        return area

    area = area_under_curve()
    col = df.columns[0]
    df[col] = df[col] / 327544.5885085
    return df


def narrow_range(df):
    """
    Restrict the spectrum to the IR fingerprint region: 630-880 cm^-1.

    This region contains the characteristic out-of-plane C-H bending bands
    that distinguish ortho-, meta-, and para-substituted benzene derivatives.

    Parameters
    ----------
    df : pd.DataFrame
        Full-range spectrum DataFrame (integer wavenumber index).

    Returns
    -------
    pd.DataFrame
        Spectrum trimmed to 630-880 cm^-1 (251 rows).
    """
    return df.loc[630:880]


def load_spectra(n, spectra_dir=None):
    """
    Load, pre-process, and combine all IR spectra from a directory.

    Reads every .txt file in spectra_dir, applies normalisation,
    interpolation, and range restriction, then concatenates them into
    a single DataFrame (251 rows x number_of_spectra columns).

    Column names are the compound name without repeat number suffix,
    so that all 5 repeats of a compound share the same column label.
    This makes train/test splitting by compound straightforward.

    Parameters
    ----------
    n : int
        Number of PCA components. Pass 0 to skip PCA and return raw spectra.
    spectra_dir : str, optional
        Path to directory of raw .txt spectra. Defaults to SPECTRA_DIR.

    Returns
    -------
    pd.DataFrame
        Combined pre-processed spectra. Shape: (251, num_spectra) if n=0,
        or (n, num_spectra) if PCA is applied.
    """
    if spectra_dir is None:
        spectra_dir = SPECTRA_DIR

    spectra = []
    for entry in os.scandir(spectra_dir):
        if not entry.name.endswith('.txt'):
            continue

        df = pd.read_csv(
            entry.path,
            skiprows=4,
            sep=r"\s+",
            names=[entry.name],
            index_col=0
        )
        df = normalisation(df)
        df = interpolation(df)
        df = narrow_range(df)

        # Strip repeat number: "p-xylene_3.txt" -> "p-xylene"
        name = entry.name.removesuffix('.txt')
        name = re.sub(r'_\d+$', '', name)
        df.columns = [name]

        spectra.append(df)

    combined = pd.concat(spectra, axis=1)

    if n > 0:
        pca_df = perform_pca(n, combined)
        print(pca_df)
        return pca_df

    return combined


def perform_pca(n, data=None, spectra_dir=None):
    """
    Apply PCA to reduce spectral dimensionality.

    sklearn expects (samples x features), so the DataFrame is transposed
    before fitting and transposed back after.

    Parameters
    ----------
    n : int
        Number of principal components to retain.
    data : pd.DataFrame, optional
        Pre-loaded spectral DataFrame. If None, load_spectra(0) is called.
    spectra_dir : str, optional
        Passed to load_spectra() if data is None.

    Returns
    -------
    pd.DataFrame
        Shape (n, num_spectra): n principal components, one column per
        spectrum, labelled by compound name.
    """
    if data is None:
        data = load_spectra(0, spectra_dir=spectra_dir)

    pca = sklearn.decomposition.PCA(n_components=n)
    pca_array = pca.fit_transform(data.T)   # fit on (spectra x wavenumbers)
    pca_retransposed = pca_array.T           # back to (components x spectra)

    pca_df = pd.DataFrame(pca_retransposed, columns=data.columns)
    return pca_df
