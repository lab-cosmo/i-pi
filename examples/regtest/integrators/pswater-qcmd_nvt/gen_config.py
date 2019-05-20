#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 13:11:04 2019

@author: gtrenins

Generates the starting configurations for testing qCMD thermal averages.
"""

import numpy as np


def export_init(r1, r2, th, outname):
    """Exports an XYZ file describing an H2O molecule with the specified values
       of OH bond-lengths and bond angle.
    """
    comment = "# CELL(abcABC): 200. 200. 200. 90. 90. 90. cell Traj: positions Step: 0 Bead: 0"
    fmt = "{:s}"+3*"  {:9.6f}"+"\n"
    qO = np.zeros(3)
    qH1 = np.zeros_like(qO)
    qH1[0] = r1  # H1 lies of X-axis
    qH2 = np.zeros_like(qO)
    qH2[0:2] = r2*np.array([np.cos(th), np.sin(th)])
    with open(outname, 'w') as f:
        f.write("{:d}\n".format(3))
        f.write("{:s}\n".format(comment))
        f.write(fmt.format("O", *qO))
        f.write(fmt.format("H", *qH1))
        f.write(fmt.format("H", *qH2))

if __name__ == "__main__":
    # 300K:
    r1 = np.array([1.500000000000000000E+00,
                   1.547619047619047672E+00,
                   1.690476190476190466E+00,
                   1.912698412698412564E+00])
    r2 = np.array([2.087301587301586991E+00,
                   1.547619047619047672E+00,
                   2.007936507936507908E+00,
                   2.214285714285714413E+00])
    th = np.array([2.181661564992912083E+00,
                   1.907395539679517249E+00,
                   1.645596151880367897E+00,
                   1.919862177193762509E+00,])
    for i in range(len(r1)):
        outname = "300K-init{:1d}.xyz".format(i)
        export_init(r1[i], r2[i], th[i], outname)
    # 150K:
    r1 = np.array([1.707142857142857073E+00,
                   1.923015873015872845E+00,
                   1.961111111111110805E+00,
                   1.999206349206348987E+00])
    r2 = np.array([1.897619047619047539E+00,
                   1.967460317460317354E+00,
                   1.713492063492063400E+00,
                   2.043650793650793496E+00])
    th = np.array([1.720395976965839013E+00,
                   1.700449356943046775E+00,
                   2.169194927478666379E+00,
                   2.099381757398893544E+00])
    for i in range(len(r1)):
        outname = "150K-init{:1d}.xyz".format(i)
        export_init(r1[i], r2[i], th[i], outname)
