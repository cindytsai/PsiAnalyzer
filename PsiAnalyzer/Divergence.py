import numpy as np


def Divergence(f_x, f_y, f_z, dx, pad):
    """
    Get Div( f ) of the vector field.

    :param f_x: x component.
    :param f_y: y component.
    :param f_z: z component.
    :param dx: Spacing between sample points.
    :param pad: Ratio of padding 0 length compare to the length of that axis at each side.
    Pad ceil( f.shape * pad ) zeros.
    :return:
    """