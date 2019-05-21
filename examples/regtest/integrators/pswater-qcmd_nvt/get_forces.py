#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 10:04:04 2019

@author: gtrenins
"""

from __future__ import print_function, division
import numpy as np

def read_frame(fobj):
    """
    Given an open file object, read a frame assuming XYZ file format,
    starting from the current position.
    """

    natom_str = fobj.readline()
    natom = int(natom_str)
    q = np.empty((natom,3))
    for i in range(natom+1):
        q_str = fobj.readline()
        if (i==0):
            continue    # skip the comment line
        q[i-1,:] = np.array(q_str.split()[1:], dtype=np.float)

    return q

def get_internals(q):
    """
    Calculate the bond lengths and angles for a triatomic.
    Args:
        q(2d array): a 3-by-3 array that holds atomic coordinates such
        that the central atom of the triatomic is placed at the start.
    """

    qOH1 = q[1,:] - q[0,:]
    qOH2 = q[2,:] - q[0,:]
    r1 = np.sqrt(np.sum(qOH1**2, axis=-1))
    r2 = np.sqrt(np.sum(qOH2**2, axis=-1))
    theta = np.arccos( np.sum(qOH1*qOH2, axis=-1)/r1/r2 )
    return r1, r2, theta

def get_forces(Q, q, f):
    """
    Calculate the contribution to the mean-field quasi-centroid force.
    Args:
        Q (2d array): 3-by-3 quasi-centroid configuration
        q (2d array): 3-by-3 bead configuration
        f (2d array): forces acting on the beads.
    """

    #R1, R2, Th = get_internals(Q)
    r1, r2, th = get_internals(q)

    q1 = q[1]-q[0]
    q1 /= r1
    q2 = q[2]-q[0]
    q2 /= r2
    fr1 = np.dot(q1, f[1])
    fr2 = np.dot(q2, f[2])
    ct = np.dot(q1, q2)
    st = np.sqrt(1.0-ct**2)
    fth = -np.dot(q2-ct*q1, f[1])*r1/st

#    F = np.zeros_like(Q)
#    Q1 = Q[1]-Q[0]
#    Q1 /= R1
#    Q2 = Q[2]-Q[0]
#    Q2 /= R2
#    CT = np.dot(Q1, Q2)
#    ST = np.sqrt(1.0-CT**2)
#    F[1] = Q1*fr1 - (Q2 - Q1*CT)*fth/(R1*ST)
#    F[2] = Q2*fr2 - (Q1 - Q2*CT)*fth/(R2*ST)
#    F[0] = -F[1]-F[2]
    return np.array([fr1, fr2, fth])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("""
Calculate the OH bond-lengths and angles, averaged over the beads, and the
functions that appear in the definition of Eckart constraints.
    """)
    parser.add_argument('ref', help="""
(Quasi-)centroid configuration used to initialise the simulation.
    """)
    parser.add_argument('beads', help="""
Trajectory files with configurations of individual bead replicas.
    """)
    parser.add_argument('forces', help="""
Trajectory files with forces on individual bead replicas.
    """)
    # Optional args
    parser.add_argument('-s', '--skip', type=int, help="""
Number of frames to be skipped at the beginning.
    """)

    args = parser.parse_args()
    with open(args.ref, 'r') as f:
        qref = read_frame(f)

    q = np.zeros_like(qref)
    f = np.zeros_like(qref)
    F = np.zeros(3)
    nsample = 0
    with open(args.beads,'r') as fq, open(args.forces,'r') as ff:
        nframe = 0
        while True:
            try:
                q[...] = read_frame(fq)
                f[...] = read_frame(ff)
            except:
                break
            nframe += 1
            if (nframe < args.skip):
                continue
            F += get_forces(qref, q, f)
            nsample += 1
    F /= nsample
    print(F)
