'''
Created on Nov 18, 2013

@author: Lee Kamentsky
'''
import numpy as np
import sys
import os
import subprocess

if __name__ == '__main__':
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    if len(sys.argv) > 3:
        algorithm = sys.argv[3]
    else:
        algorithm = "SURF"
    
    d = {}
    for root, dirnames, filenames in os.walk(in_path):
        rroot, folder = os.path.split(root)
        c1 = [f for f in filenames if f.lower().endswith("_c1.tif")]
        if len(c1) > 0 and folder.isdigit():
            if not rroot in d.keys():
                d[rroot] = []
            d[rroot].append((int(folder), c1[0]))
    
    for root in d.keys():
        folders = sorted(d[root])
        for (f1, fn1), (f2, fn2) in zip(folders[:-1], folders[1:]):
            path1 = os.path.join(root, unicode(f1), fn1)
            path2 = os.path.join(root, unicode(f2), fn2)
            out_folder = os.path.join(out_path, os.path.split(root)[1])
            if not os.path.isdir(out_folder):
                os.makedirs(out_folder)
            pathout = os.path.join(out_folder, "%d_%d.tif" % (f1, f2))
            subprocess.check_call([sys.executable, "align.py", "--algorithm=%s" % algorithm, path1, path2, pathout])
