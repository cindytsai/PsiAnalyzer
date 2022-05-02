import numpy as np
import matplotlib.pyplot as plt
from .Gradient import Gradient_with_fft


def GetVelocity(dataRe, dataIm, cell_space, fft_pad, check_convergence=False, check_pad=[]):
    """
    Get velocity field of psi.

    :param dataRe: Re( psi )
    :param dataIm: Im( psi )
    :param cell_space: Spacing between sample points.
    :param fft_pad: Ratio of padding width to target field width at each side.
    :param check_convergence: Check convergence. Default is False.
    :param check_pad: List of padding ratio to check for convergence. If fft_pad does not include in the list, it will
    be checked as well. This is only for convergence check, the output will still be evaluated using fft_pad.
    :return: List [v_x, v_y, v_z], v_x/y/z is velocity field with shape same as input array.
    """
    # get grad(dataRe) and grad(dataIm) through fft method
    # plot difference of paddings in list check_pad, if check_convergence == True.
    if check_convergence is True:
        diff_Re = [[], [], []]  # difference average of previous padding to this padding in x,y,z respectively.
        diff_Im = [[], [], []]
        flag, first_enter = True, True
        num_samples = dataRe.shape[0] * dataRe.shape[1] * dataRe.shape[0]
        if fft_pad not in check_pad:
            check_pad.append(fft_pad)
        check_pad.sort()
        for pad in check_pad:
            # save result in data1/2 respectively, so that we can compare.
            if flag is True:
                dataRe1 = Gradient_with_fft(dataRe, cell_space, pad)
                dataIm1 = Gradient_with_fft(dataIm, cell_space, pad)
                flag = False
            else:
                dataRe2 = Gradient_with_fft(dataRe, cell_space, pad)
                dataIm2 = Gradient_with_fft(dataIm, cell_space, pad)
                flag = True
            if first_enter is False:
                for d in range(3):
                    diff_Re[d].append(np.sum(np.absolute(dataRe1[d] - dataRe2[d])) / num_samples)
                    diff_Im[d].append(np.sum(np.absolute(dataIm1[d] - dataIm2[d])) / num_samples)
            first_enter = False

            # load fft_pad result inside grad_dataRe/Im_x/y/z, this is the padding evaluated for the output.
            if pad == fft_pad:
                if flag is False:
                    grad_dataRe_x, grad_dataRe_y, grad_dataRe_z = dataRe1
                    grad_dataIm_x, grad_dataIm_y, grad_dataIm_z = dataIm1
                else:
                    grad_dataRe_x, grad_dataRe_y, grad_dataRe_z = dataRe2
                    grad_dataIm_x, grad_dataIm_y, grad_dataIm_z = dataIm2

        # plot the result.
        fig, ax = plt.subplots(2, 3)
        fig.suptitle("Difference Between Paddings")

        x_tick = np.array(check_pad)[1:]
        x_min, x_max = np.min(np.array(check_pad)), np.max(np.array(check_pad))
        subtitleRe = ["Re_x", "Re_y", "Re_z"]
        subtitleIm = ["Im_x", "Im_y", "Im_z"]
        for d in range(3):
            ax[0, d].plot(x_tick, np.array(diff_Re[d]), '.-')
            ax[1, d].plot(x_tick, np.array(diff_Im[d]), '.-')
            ax[0, d].set_title(subtitleRe[d])
            ax[1, d].set_title(subtitleIm[d])
            ax[0, d].set_xlim([x_min, x_max])
            ax[1, d].set_xlim([x_min, x_max])
        fig.savefig('check_velocity_convergence.png', bbox_inches="tight")

    else:
        grad_dataRe_x, grad_dataRe_y, grad_dataRe_z = Gradient_with_fft(dataRe, cell_space, fft_pad)
        grad_dataIm_x, grad_dataIm_y, grad_dataIm_z = Gradient_with_fft(dataIm, cell_space, fft_pad)

    # get probability current.
    J_x = -dataIm * grad_dataRe_x + dataRe * grad_dataIm_x
    J_y = -dataIm * grad_dataRe_y + dataRe * grad_dataIm_y
    J_z = -dataIm * grad_dataRe_z + dataRe * grad_dataIm_z

    # divide density at each sample points.
    density = np.power(dataRe, 2) + np.power(dataIm, 2)

    return J_x / density, J_y / density, J_z / density
