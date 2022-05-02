import numpy as np
import math


def Gradient_with_fft(f, dx, pad):
    """
    Get gradient of function f through FFT method.
    1. Pad 0.0 outside the region.
    2. Compute FFT, then times i.k.FFT[f] to get FFT[grad(f)]
    3. Compute IFFT to get grad(f).

    :param f: Target function, in [x][y][z] orientation.
    :param pad: Ratio of padding 0 length compare to the length of that axis at each side.
    Pad ceil( f.shape * pad ) zeros.
    :return: List [fx, fy, fz]
    """
    # padding 0 outside
    pad_x, pad_y, pad_z = math.ceil(pad * f.shape[0]), math.ceil(pad * f.shape[1]), math.ceil(pad * f.shape[2])
    pad_array = np.zeros((2 * pad_x, f.shape[1], f.shape[2]))
    f_pad = np.concatenate((f, pad_array), axis=0)
    pad_array = np.zeros((f_pad.shape[0], 2 * pad_y, f.shape[2]))
    f_pad = np.concatenate((f_pad, pad_array), axis=1)
    pad_array = np.zeros((f_pad.shape[0], f_pad.shape[1], 2 * pad_z))
    f_pad = np.concatenate((f_pad, pad_array), axis=2)
    f_pad = np.roll(f_pad, (pad_x, pad_y, pad_z), axis=(0, 1, 2))

    # do FFT to f_pad
    f_k = np.fft.rfftn(f_pad)

    # compute i.k.f_pad to get grad(f) in k-space.
    k_x = np.fft.fftfreq(f_pad.shape[0], d=dx)
    k_y = np.fft.fftfreq(f_pad.shape[1], d=dx)
    k_z = np.fft.rfftfreq(f_pad.shape[2], d=dx)
    k_xx, k_yy, k_zz = [2.0 * np.pi * matrix for matrix in np.meshgrid(k_x, k_y, k_z, indexing='ij')]

    grad_f_kx = 1j * k_xx * f_k
    grad_f_ky = 1j * k_yy * f_k
    grad_f_kz = 1j * k_zz * f_k

    # do inverse FFT, and dig out the original region.
    grad_fx = np.fft.irfftn(grad_f_kx)
    grad_fy = np.fft.irfftn(grad_f_ky)
    grad_fz = np.fft.irfftn(grad_f_kz)

    grad_fx = np.roll(grad_fx, (-pad_x, -pad_y, -pad_z), axis=(0, 1, 2))[0:f.shape[0], 0:f.shape[1], 0:f.shape[2]]
    grad_fy = np.roll(grad_fy, (-pad_x, -pad_y, -pad_z), axis=(0, 1, 2))[0:f.shape[0], 0:f.shape[1], 0:f.shape[2]]
    grad_fz = np.roll(grad_fz, (-pad_x, -pad_y, -pad_z), axis=(0, 1, 2))[0:f.shape[0], 0:f.shape[1], 0:f.shape[2]]

    return [grad_fx, grad_fy, grad_fz]

def Gradient_with_numpy(f):
    """
    Get gradient of function f through np.gradient.

    :param f: Target function.
    :return: List [fx, fy, fz]
    """
    return np.gradient(f)
