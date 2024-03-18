import math

import numpy as np

from get_ders import *


def solarradiation_DDR(dem, lat, cs, d, alpha):
    # d = 272s
    d = int(d)
    r = 0.2
    S0 = 1367
    tau_a = 365
    dr = 0.0174532925

    # convert factors
    slop, asp = get_ders(dem, cs)
    dummy, L = np.meshgrid(np.arange(0, len(dem[1] + 1)), lat)
    dummy = 0
    del (dummy)

    L = L * dr
    fcirc = 360 * dr

    sinL = np.sin(L)
    cosL = np.cos(L)
    tanL = np.tan(L)
    sinSlop = np.sin(slop)
    cosSlop = np.cos(slop)
    cosSlop2 = cosSlop * cosSlop
    sinSlop2 = sinSlop * sinSlop
    sinAsp = np.sin(asp)
    cosAsp = np.cos(asp)
    term1 = (sinL * cosSlop - cosL * sinSlop * cosAsp)
    term2 = (cosL * cosSlop + sinL * sinSlop * cosAsp)
    term3 = sinSlop * sinAsp

    # alpha = 0.766917695568954  # np.radians(alpha)
    alpha = float(alpha)
    alpha = np.deg2rad(alpha)

    # print(alpha)
    I0 = S0 * (1 + 0.0344 * np.cos(fcirc * d / tau_a))
    dS = 23.45 * dr * np.sin(fcirc * ((284 + d) / tau_a))
    hsr = np.real(np.arccos(-tanL * np.tan(dS)))
    It = round(12 * (1 + np.average(np.array(hsr).reshape(-1, 1)) / math.pi) - 12 * (
            1 - np.average(np.array(hsr).reshape(-1, 1)) / math.pi))
    hs = np.arccos((np.sin(alpha) - sinL * np.sin(dS)) / (cosL * np.cos(dS)))
    row, col = np.shape(dem)
    sinAlpha = np.sin(alpha)
    M = np.sqrt(1229 + ((614 * sinAlpha)) ** 2) - 614 * sinAlpha
    tau_b = 0.56 * (np.exp(-0.65 * M) + np.exp(-0.095 * M))
    tau_d = 0.271 - 0.294 * tau_b
    tau_r = 0.271 + 0.706 * tau_b
    cos_i = (np.sin(dS) * term1) + (np.cos(dS) * np.cos(hs) * term2) + (np.cos(dS) * term3 * np.sin(hs))

    Is = I0 * tau_b
    R = Is * cos_i

    # for i in R:
    #     if i < 0:
    #         i = 0
    R = np.where(R < 0, 0, R)
    Id = I0 * tau_d * cosSlop2 / 2 * sinAlpha
    Ir = I0 * r * tau_r * sinSlop2 / 2 * sinAlpha

#     print("R : ",R)
    # print(Id[0, 0])
    # print(Ir[0, 0])

    return [R, Id, Ir]
