import numpy as np
import pandas as pd
from PsiAnalyzer.Normalized import Normalized

## Basic Data Info.
filepath = "./Data/"
fieldpath = {"Real": "test_wf_1.00e12_AMR_resolution_factor_10_Real_Level=3_UM_IC.bin",
             "Imag": "test_wf_1.00e12_AMR_resolution_factor_10_Imag_Level=3_UM_IC.bin"}
dimensions = [480, 480, 480]
center_idx = [239.5, 239.5, 239.5]
datatype = np.float32

code_unit = 3.43381323e-05
cell_unit = 4.58351745817846e+24    # "cm"
density_unit = 2.57585797031067e-30 # "g/cm^3"

profile_filename = "profile_Data_000000.txt"


## Read from file, then reshape and swapaxes to [x][y][z] coordinate
field = {"Real": np.fromfile(filepath+fieldpath["Real"], datatype).reshape(dimensions).swapaxes(0, 2),
         "Imag": np.fromfile(filepath+fieldpath["Imag"], datatype).reshape(dimensions).swapaxes(0, 2)}

kpc2cm = 3.086e+21
profile_raw = pd.read_csv(filepath + profile_filename, skiprows=1, names=["Radius", "Density"], sep=" ")
profile = {"Radius": profile_raw["Radius"].to_numpy() * kpc2cm,
           "Density": profile_raw["Density"].to_numpy()}

## (1) Get normalized Re(psi) and Im(psi) field
NormField = dict()

# Calculate using Normalized function.
NormField["Real"], NormField["Imag"] = Normalized(profile, field["Real"], field["Imag"], dimensions, datatype,
                                                  center_idx, code_unit * cell_unit, density_unit)
np.savez(filepath+"NormField.npz", Real=NormField["Real"], Imag=NormField["Imag"])

# Read from file.
# NormField["Real"] = np.fromfile(filepath+"NormReal.bin", datatype).reshape(dimensions)
# NormField["Imag"] = np.fromfile(filepath+"NormImag.bin", datatype).reshape(dimensions)

## (2) Do DFT
