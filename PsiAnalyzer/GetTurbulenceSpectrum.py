import numpy as np
import matplotlib.pyplot as plt
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


def GetTurbulenceSpectrum(v_x, v_y, v_z, cell_space, fft_pad, num_bin=200, check_convergence=False, check_pad=[]):
    if check_convergence is True:
        if fft_pad not in check_pad:
            check_pad.append(fft_pad)
        check_pad.sort()
    else:
        check_pad.clear()
        check_pad.append(fft_pad)

    # find turbulence spectrum with different paddings.
    for pad in check_pad:
        # pad zero outside target region, find fft(v_x/y/z), and add them together.
        v_k = np.power(np.absolute(np.fft.rfftn(_pad_zeros(v_x, pad))), 2) + \
              np.power(np.absolute(np.fft.rfftn(_pad_zeros(v_y, pad))), 2) + \
              np.power(np.absolute(np.fft.rfftn(_pad_zeros(v_z, pad))), 2)
        v_k_shape = [math.ceil(pad * v_x.shape[0]) * 2 + v_x.shape[0],
                     math.ceil(pad * v_x.shape[1]) * 2 + v_x.shape[1],
                     math.ceil(pad * v_x.shape[2]) * 2 + v_x.shape[2]]

        # create k = (kx, ky, kz)
        kx2 = np.power(np.fft.fftfreq(v_k_shape[0], d=cell_space), 2)
        ky2 = np.power(np.fft.fftfreq(v_k_shape[1], d=cell_space), 2)
        kz2 = np.power(np.fft.rfftfreq(v_k_shape[2], d=cell_space), 2)
        xx, yy, zz = np.meshgrid(kx2, ky2, kz2, indexing='ij')
        kk_norm = np.sqrt(xx + yy + zz)

        k_min, k_max = np.sqrt(kx2[1]), (1.0 + 1e-6) * np.sqrt(np.max(kx2) + np.max(ky2) + np.max(kz2))
        k_norm = np.logspace(np.log10(k_min), np.log10(k_max), num=num_bin+1, endpoint=True)
        k_shift = 10 ** ((np.log10(k_max) - np.log10(k_min)) / (num_bin * 2.0))
        del kx2
        del ky2
        del kz2
        del xx
        del yy
        del zz

        # binning k and do average.
        power_spectrum = []
        for i in range(num_bin):
            mask = np.logical_and(k_norm[i] <= kk_norm, kk_norm < k_norm[i+1])
            with np.errstate(divide='ignore'):
                power_spectrum.append(np.sum(mask * v_k) / np.sum(mask))

        # add to image buffer, and shift k_norm.
        plt.plot(k_norm[:-1] * k_shift, power_spectrum, '.-', label='pad ratio = {}'.format(pad))

    # Plot the result
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.title('Turbulence Spectrum')
    plt.xlabel(r'$\log k$')
    plt.ylabel(r'$\log E$')
    plt.savefig('TurbulenceSpectrum.png')
