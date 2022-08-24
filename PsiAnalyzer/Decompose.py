import numpy as np
import math


def _pad_zeros(f, pad):
    pad_x, pad_y, pad_z = math.ceil(pad * f.shape[0]), math.ceil(pad * f.shape[1]), math.ceil(pad * f.shape[2])
    pad_array = np.zeros((2 * pad_x, f.shape[1], f.shape[2]))
    f_pad = np.concatenate((f, pad_array), axis=0)
    pad_array = np.zeros((f_pad.shape[0], 2 * pad_y, f.shape[2]))
    f_pad = np.concatenate((f_pad, pad_array), axis=1)
    pad_array = np.zeros((f_pad.shape[0], f_pad.shape[1], 2 * pad_z))
    f_pad = np.concatenate((f_pad, pad_array), axis=2)
    f_pad = np.roll(f_pad, (pad_x, pad_y, pad_z), axis=(0, 1, 2))
    return f_pad


def Decompose(v_x, v_y, v_z, cell_space, pad):
    """
    Decompose input 3-dim vector field to longitudinal and rotational vector field.

    :param v_x: vector field x-component.
    :param v_y: vector field y-component.
    :param v_z: vector field z-component.
    :param cell_space: sampling space.
    :param pad: padding zero ratio outside the target region.
    :return: [v_long_x, v_long_y, v_long_z], [v_rota_x, v_rota_y, v_rota_z]
    """

    # pad zeros
    v_x_pad = _pad_zeros(v_x, pad)
    v_y_pad = _pad_zeros(v_y, pad)
    v_z_pad = _pad_zeros(v_z, pad)
    pad_x, pad_y, pad_z = math.ceil(pad * v_x.shape[0]), math.ceil(pad * v_x.shape[1]), math.ceil(pad * v_x.shape[2])

    # find phi_k, and replace nan with 0.
    k_x = np.fft.fftfreq(v_x_pad.shape[0], d=cell_space)
    k_y = np.fft.fftfreq(v_x_pad.shape[1], d=cell_space)
    k_z = np.fft.rfftfreq(v_x_pad.shape[2], d=cell_space)
    k_xx, k_yy, k_zz = [2.0 * np.pi * matrix for matrix in np.meshgrid(k_x, k_y, k_z, indexing='ij')]

    phi_k = -1.0j * (k_xx * np.fft.rfftn(v_x_pad) + k_yy * np.fft.rfftn(v_y_pad) + k_zz * np.fft.rfftn(v_z_pad))
    phi_k = phi_k / (np.power(k_xx, 2) + np.power(k_yy, 2) + np.power(k_zz, 2))
    phi_k[np.isnan(phi_k)] = 0.0

    # find longitudinal and rotational component.
    v_long_x = np.fft.irfftn(1.0j * k_xx * phi_k)
    v_long_y = np.fft.irfftn(1.0j * k_yy * phi_k)
    v_long_z = np.fft.irfftn(1.0j * k_zz * phi_k)

    v_long_x = np.roll(v_long_x, (-pad_x, -pad_y, -pad_z), axis=(0, 1, 2))[0:v_x.shape[0], 0:v_x.shape[1], 0:v_x.shape[2]]
    v_long_y = np.roll(v_long_y, (-pad_x, -pad_y, -pad_z), axis=(0, 1, 2))[0:v_y.shape[0], 0:v_y.shape[1], 0:v_y.shape[2]]
    v_long_z = np.roll(v_long_z, (-pad_x, -pad_y, -pad_z), axis=(0, 1, 2))[0:v_z.shape[0], 0:v_z.shape[1], 0:v_z.shape[2]]

    v_rota_x = v_x - v_long_x
    v_rota_y = v_y - v_long_y
    v_rota_z = v_z - v_long_z

    return [v_long_x, v_long_y, v_long_z], [v_rota_x, v_rota_y, v_rota_z]


def CheckDecomposeRota(v_rota_x, v_rota_y, v_rota_z, pad):
    f_size = v_rota_x.shape
    v_rota_x = _pad_zeros(v_rota_x, pad)
    v_rota_y = _pad_zeros(v_rota_y, pad)
    v_rota_z = _pad_zeros(v_rota_z, pad)
    pad_x, pad_y, pad_z = math.ceil(pad * f_size[0]), math.ceil(pad * f_size[1]), math.ceil(pad * f_size[2])

