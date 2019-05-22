#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 16:55:40 2019

@author: gtrenins
"""

import os
import re
import subprocess

def local(file=None):
    """Returns local folder of the tests directory.

    Args:
        - file: Append file to the local folder
    """
    if file is None:
        return os.path.abspath(os.path.dirname(__file__))
    else:
        return os.path.join(os.path.dirname(__file__), file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("""
Submit a test job to run on odyssey using the dummy input files script.in and
input.xml.in
    """)
    parser.add_argument("dir", help="""
Directory from which to run the job.
""")
    parser.add_argument('-T', '--temp', type=int, help="""
Similation temperature in K.
    """)
    parser.add_argument('-n', '--nbeads', type=int, help="""
Number of beads.
    """)
    parser.add_argument('-s', '--stride', type=int, help="""
Print forces and positions for every 'stride' beads.
    """)
    parser.add_argument('-p', '--port', type=int, help="""
Port number (integer between 1025 and 65535).
    """)
    parser.add_argument('-l', '--split', choices=['baoab','obabo'], help="""
Choose the splitting.
    """)

    args = parser.parse_args()
    root = local()
    target = local(args.dir)
    tags = [ r'@PORT@', r'@NB@', r'@SB@', r'@T@', r'@SPLIT@']
    vals = [r'{:d}'.format(val) for val in [args.port, args.nbeads,
                                            args.stride, args.temp]]
    vals.append(args.split)
    with open("input.xml.in", 'r') as fin, open("input.xml", "w") as fout:
        for line in fin:
            newline = line
            for tag, val in zip(tags, vals):
                newline = re.sub(tag, val, newline)
            fout.write(newline)
    os.rename("input.xml", os.path.join(target, "input.xml"))
    with open("script.in", 'r') as fin, open("script", "w") as fout:
        for line in fin:
            newline = line
            for tag, val in zip(tags[:2], vals[:2]):
                newline = re.sub(tag, val, newline)
            fout.write(newline)
    os.rename("script", os.path.join(target, "script"))
    os.chdir(target)
    cmd = "qsub script".split()
    subprocess.call(cmd)
    os.chdir(root)


