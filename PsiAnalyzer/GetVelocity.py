import numpy as np
import matplotlib.pyplot as plt
import Gradient


def GetVelocity(dataRe, dataIm, cell_space, fft_pad, check_convergence=False):
    # get grad(dataRe) and grad(dataIm) through fft method
    # plot difference of paddings from 1~fft_pad, if check_convergence == True.
    # TODO: See if padding ratio alters from 0~fft_pad, will result converges. And plot diff from each paddings.
    if check_convergence is True:
        diff_Re = [[], [], []]  # difference average of previous padding to this padding in x,y,z respectively.
        diff_Im = [[], [], []]
        flag = True
        num_samples = dataRe.shape[0] * dataRe.shape[1] * dataRe.shape[0]
        for pad in range(fft_pad + 1):
            if flag is True:
                dataRe1 = Gradient.Gradient_with_fft(dataRe, cell_space, pad)
                dataIm1 = Gradient.Gradient_with_fft(dataIm, cell_space, pad)
                flag = False
            else:
                dataRe2 = Gradient.Gradient_with_fft(dataRe, cell_space, pad)
                dataIm2 = Gradient.Gradient_with_fft(dataIm, cell_space, pad)
                flag = True
            if pad != 0:
                for d in range(3):
                    diff_Re[d].append(np.sum(np.absolute(dataRe1[d] - dataRe2[d])) / num_samples)
                    diff_Im[d].append(np.sum(np.absolute(dataIm1[d] - dataIm2[d])) / num_samples)

        # TODO:START HERE. Plot the result.

        # load grad_dataRe_x/y/z, and grad_dataIm_x/y/z
        if flag is False:
            grad_dataRe_x, grad_dataRe_y, grad_dataRe_z = dataRe1
            grad_dataIm_x, grad_dataIm_y, grad_dataIm_z = dataIm1
        else:
            grad_dataRe_x, grad_dataRe_y, grad_dataRe_z = dataRe2
            grad_dataIm_x, grad_dataIm_y, grad_dataIm_z = dataIm2

    else:
        grad_dataRe_x, grad_dataRe_y, grad_dataRe_z = Gradient.Gradient_with_fft(dataRe, cell_space, fft_pad)
        grad_dataIm_x, grad_dataIm_y, grad_dataIm_z = Gradient.Gradient_with_fft(dataIm, cell_space, fft_pad)

    # get probability current.
    J_x = -dataIm * grad_dataRe_x + dataRe * grad_dataIm_x
    J_y = -dataIm * grad_dataRe_y + dataRe * grad_dataIm_y
    J_z = -dataIm * grad_dataRe_z + dataRe * grad_dataIm_z

    # divide density at each sample points.
    density = np.power(dataRe, 2) + np.power(dataIm, 2)

    return J_x / density, J_y / density, J_z / density
