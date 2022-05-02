import numpy


def GetVelocity_Longitudinal(V_x, V_y, V_z, cell_space, fft_pad, check_convergence=False, check_pad=[]):
    """
    Extract longitudinal component in velocity field.

    :param V_x: Velocity x-component.
    :param V_y: Velocity y-component.
    :param V_z: Velocity z-component.
    :param cell_space: Spacing between sample points.
    :param fft_pad: Ratio of padding width to target field width at each side.
    :param check_convergence: Check convergence. Default is False.
    :param check_pad: List of padding ratio to check for convergence. If fft_pad does not include in the list, it will
    be checked as well. This is only for convergence check, the output will still be evaluated using fft_pad.
    :return: List [Vlong_x, Vlong_y, Vlong_z]
    """
    pass