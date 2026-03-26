import numpy as np 
import pandas as pd 
import os
from google.colab import drive
import sklearn.decomposition
drive.mount('/content/drive')
import re
import importlib

# Interpolation 
def interpolation(df_1):
  index_list = list(df_1.index)
  minimum_index = round(min(index_list))
  maximum_index = round(max(index_list))
  list1 = list(range(minimum_index, maximum_index+1))
  list2 = []
  for i in list1:
    list2 = np.append(list2, np.nan)
  # df_2 = pd.DataFrame(data=list2, index=list1, columns=["% Transmittance"])
  df_2 = pd.DataFrame(
    data=list2,
    index=list1,
    columns=[df_1.columns[0]]
)
  df = pd.concat([df_1,df_2])
  df = df.sort_index()
  df = df.interpolate()
  df = df.loc[df.index.isin(list1)]

  return df

# Normalisation
def normalisation(df):
  def area_under_curve():
    x_values = df.index
    y_values = df.values
    area = (x_values, y_values)
    area = 0.0
    for i in range(1, len(x_values)):
        area += 0.5 * (y_values[i] + y_values[i-1]) * (x_values[i] - x_values[i-1])
    return area
  area = area_under_curve()
  col = df.columns[0]
  df[col] = df[col]/327544.5885085 
  return df

# Narrowing the wavenumber range
def narrow_range(df):
  df = df.loc[630:880]
  return df

# Load spectra (5b)
def load_spectra_0():
  os.scandir("/content/drive/MyDrive/Raw_IR_Spectra")
  c = []
  for i in os.scandir("/content/drive/MyDrive/Raw_IR_Spectra"):
    df = pd.read_csv(i,
                     skiprows=4,
                     sep=r"\s+",
                     names=[i.name],
                     index_col=0)
    df = normalisation(df)
    df = interpolation(df)
    df = narrow_range(df)
    c.append(df)
  c = pd.concat(c, axis=1)
  return c

# Perform PCA (6)
def perform_pca(n):
  d = load_spectra_0()
  pca = sklearn.decomposition.PCA(n_components = n)
  pca_transposed = d.T
  pca_array = pca.fit_transform(pca_transposed)
  pca_retransposed = pca_array.T
  column_names = []
  for i in os.scandir("/content/drive/MyDrive/Raw_IR_Spectra"):
    name = i.name.removesuffix('.txt')
    name = re.sub(r'_\d+$', '', name)
    column_names.append(name)
  pca_df = pd.DataFrame(pca_retransposed, columns=column_names)
  return pca_df

# Load spectra (6 edited version)
def load_spectra(n):
  c = []
  columns_identical = []
  for i in os.scandir("/content/drive/MyDrive/Raw_IR_Spectra"):
    df = pd.read_csv(i,
                     skiprows=4,
                     sep=r"\s+",
                     names=[i.name],
                     index_col=0)
    df = normalisation(df)
    df = interpolation(df)
    df = narrow_range(df)
    name = i.name.removesuffix('.txt')
    name = re.sub(r'_\d+$', '', name)
    df.columns = [name]
    c.append(df)
  c = pd.concat(c, axis=1)
  if n > 0:
    a = perform_pca(n)
    print(a)
  else:
    print(c)
  return c
