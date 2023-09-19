#!/usr/bin/false
# Vectorized versions of rgb_to_hls and hls_to_rgb from the colorsys library

import numpy as np

def vector_rgb_to_hls(colors):
    r = colors[:,0]
    g = colors[:,1]
    b = colors[:,2]
    maxc = colors.max(axis=1)
    minc = colors.min(axis=1)
    sumc = (maxc+minc)
    rangec = (maxc-minc)
    l = sumc/2.0
    s = np.where(l <= 0.5, rangec / sumc, rangec / (2.0-maxc-minc))
    with np.errstate(divide="ignore", invalid="ignore"):
        rc = (maxc-r) / rangec
        gc = (maxc-g) / rangec
        bc = (maxc-b) / rangec
    h = np.where(r == maxc,
                 bc-gc,
                 np.where(g == maxc,
                          2.0+rc-bc,
                          4.0+gc-rc))
    h = (h/6.0) % 1.0
    
    fh = np.where(rangec == 0, 0.0, h)
    fl = l
    fs = np.where(rangec == 0, 0.0, s)
    return np.stack((fh, fl, fs), axis=1)

def vector_hls_to_rgb(colors):
    h = colors[:,0]
    l = colors[:,1]
    s = colors[:,2]
    m2 = np.where(l <= 0.5, l * (1.0+s), l+s-(l*s))
    m1 = 2.0*l - m2
    r = np.where(s == 0.0, l, _v(m1, m2, h+(1/3)))
    g = np.where(s == 0.0, l, _v(m1, m2, h))
    b = np.where(s == 0.0, l, _v(m1, m2, h-(1/3)))
    return np.stack((r, g, b), axis=1)

def _v(m1, m2, hue):
    hue = hue % 1.0

    return np.where(hue < (1/6),
                    m1 + (m2-m1)*hue*6.0,
                    np.where(hue < (1/2),
                             m2,
                             np.where(hue < (2/3),
                                      m1 + (m2-m1)*((2/3)-hue)*6.0,
                                      m1)))
