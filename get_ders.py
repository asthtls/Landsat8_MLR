import math

import numpy as np


def get_ders(dem, cs):
    y, x = np.gradient(dem, cs, cs)  # uses simple, unweighted gradient of immediate neighbours

    def cart2pol(x, y):
        rho = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        return (rho, phi)

    grad, asp = cart2pol(y, x)
    grad = np.arctan(grad)
    asp = asp * -1 + math.pi

    return [grad, asp]
