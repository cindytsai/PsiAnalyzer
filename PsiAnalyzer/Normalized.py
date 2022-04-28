import numpy as np

def Normalized(profile, dataRe, dataIm, data_dim, data_type, center_idx, cell_space, density_unit):
    """
    This function will normalize real and imaginary part of psi to flat field.
    It does not have the conception of unit. Check it before sending in.
    Which means after normalization, Re(psi)^2 + Im(psi)^2 is a flat density field.
    :param profile: A dict contains keys "Radius", "Density".
    :param dataRe: Real field to normalize.
    :param dataIm: Imag field to normalize.
    :param data_type: NumPy data type.
    :param data_dim: data's dimension, in [x][y][z] orientation.
    :param center_idx: Center index of the data. It doesn't necessarily need to be an integer.
    :param cell_space: Size of the cell in "cm".
    :param density_unit: Density unit of psi field.
    :return: flat normalized Re(psi), Im(psi)
    """
    # Read profile
    ProRadius = profile["Radius"]
    ProDensity = profile["Density"]

    # Calculate each cell distance from center index
    X, Y, Z = np.meshgrid(np.arange(data_dim[0]), np.arange(data_dim[1]), np.arange(data_dim[2]), indexing='ij')
    X = X - center_idx[0]
    Y = Y - center_idx[1]
    Z = Z - center_idx[2]
    RadiusData = np.sqrt(np.power(X, 2) + np.power(Y, 2) + np.power(Z, 2)) * cell_space

    # Build matrix to do interpolate.
    R1 = np.zeros(data_dim, dtype=data_type)
    R2 = np.zeros(data_dim, dtype=data_type)
    D1 = np.zeros(data_dim, dtype=data_type)
    D2 = np.zeros(data_dim, dtype=data_type)
    for i in range(len(ProRadius) - 1):
        mask = (ProRadius[i] <= RadiusData) & (RadiusData < ProRadius[i+1])
        R1 += ProRadius[i] * mask
        R2 += ProRadius[i+1] * mask
        D1 += ProDensity[i] * mask
        D2 += ProDensity[i+1] * mask

    # Fill in density directly if radius smaller or bigger than the profile radius.
    mask = (RadiusData < ProRadius[0])
    RadiusData = ~mask * RadiusData
    RadiusData += mask * ProRadius[0]
    R1 += ProRadius[0] * mask
    R2 += ProRadius[1] * mask
    D1 += ProDensity[0] * mask
    D2 += ProDensity[1] * mask

    mask = (ProRadius[-1] <= RadiusData)
    RadiusData = ~mask * RadiusData
    RadiusData += mask * ProRadius[-1]
    R1 += ProRadius[-2] * mask
    R2 += ProRadius[-1] * mask
    D1 += ProDensity[-2] * mask
    D2 += ProDensity[-1] * mask

    assert (len(np.argwhere(R2 == 0)) == 0) or (len(np.argwhere(R1 == 0)) == 0), "Matrix element for interpolating " \
                                                                                 "not set. "
    AveDensity = ((R2 - RadiusData)*D1 + (RadiusData - R1)*D2) / (R2 - R1)

    # Return normalized real and imaginary part of psi.
    return dataRe * np.sqrt(density_unit / AveDensity), dataIm * np.sqrt(density_unit / AveDensity)
