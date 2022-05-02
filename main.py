import numpy as np
import pandas as pd
from PsiAnalyzer.Normalize import Normalize
from PsiAnalyzer.GetVelocity import GetVelocity

## Read data file name and basic info.
filepath = "./Data/"
fieldpath = {"Real": "test_wf_1.00e12_AMR_resolution_factor_10_Real_Level=3_UM_IC.bin",
             "Imag": "test_wf_1.00e12_AMR_resolution_factor_10_Imag_Level=3_UM_IC.bin"}  # [z][y][x] coordinate

datatype = np.float32
dimensions = [480, 480, 480]
center_idx = [239.5, 239.5, 239.5]
code_unit = 3.43381323e-05
cell_unit = 4.58351745817846e+24  # "cm"
density_unit = 2.57585797031067e-30  # "g/cm^3"

## Read profile file name.
profile_filename = "profile_Data_000000.txt"  # Radius(kpc), Density(g/cm^3)

## Read from file, then reshape and swapaxes to [x][y][z] coordinate
field = {"Real": np.fromfile(filepath + fieldpath["Real"], datatype).reshape(dimensions).swapaxes(0, 2),
         "Imag": np.fromfile(filepath + fieldpath["Imag"], datatype).reshape(dimensions).swapaxes(0, 2)}

kpc2cm = 3.086e+21
profile_raw = pd.read_csv(filepath + profile_filename, skiprows=1, names=["Radius", "Density"], sep=" ")
profile = {"Radius": profile_raw["Radius"].to_numpy() * kpc2cm,
           "Density": profile_raw["Density"].to_numpy()}


## (1) Get normalized Re(psi) and Im(psi) field
# NormField = dict()

##     Calculate using Normalized function.
# NormField["Real"], NormField["Imag"] = Normalize(profile, field["Real"], field["Imag"], dimensions, datatype,
#                                                  center_idx, code_unit * cell_unit, density_unit)
# np.savez(filepath + "NormField.npz", Real=NormField["Real"], Imag=NormField["Imag"])

##     Read from file.
# temp = np.load(filepath + "NormField.npz")
# NormField["Real"] = temp["Real"]
# NormField["Imag"] = temp["Imag"]

## (2) Get velocity field
VelField = dict()

##     Calculate using GetVelocity function, using FFT methods.
# VelField["VelX"], VelField["VelY"], VelField["VelZ"] = GetVelocity(NormField["Real"], NormField["Imag"],
#                                                                    code_unit*cell_unit, 0.2, check_convergence=True,
#                                                                    check_pad=[0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175])
# np.savez(filepath + "VelocityField.npz", VelX=VelField["VelX"], VelY=VelField["VelY"], VelZ=VelField["VelZ"])

##     Read from file
temp = np.load(filepath + "VelocityField.npz")
VelField["VelX"] = temp["VelX"]
VelField["VelY"] = temp["VelY"]
VelField["VelZ"] = temp["VelZ"]

## (3) Power Spectrum

