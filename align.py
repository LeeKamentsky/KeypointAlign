'''
Created on Nov 15, 2013

@author: Lee Kamentsky
'''
import hashlib
import os
import matplotlib
import matplotlib.cm
import matplotlib.image
import matplotlib.backends.backend_pdf
import numpy as np
from scipy.linalg import svd, det
from scipy.ndimage import map_coordinates
import cv2
import sys

N_MATCHES = 30
N_GROUPS = 200
N_PER_GROUP = 5
ERROR_EPS = 5.0
GOOD_VOTE_COUNT = 15
SIFT_SIGMA = 5
MIN_KP_DISTANCE = 5
assert hasattr(matplotlib.cm, "gray")

def compute_alignment(pt1, pt2):
    '''Compute an offset and rotation given a set of point correspondences
    
    pt1, pt2 - N vectors of D coordinates each
    
    returns an offset vector from pt1 to pt2 of length D and
    a DxD rotation matrix.
    '''
    p = np.mean(pt1, 0)
    q = np.mean(pt2, 0)
    pt1n = pt1 - p[np.newaxis, :]
    pt2n = pt2 - q[np.newaxis, :]
    c = np.dot(pt1n.transpose(), pt2n)
    u, s, v = svd(c)
    rot = np.dot(u, v.transpose())
    off = p - np.dot(rot, q)
    return off, rot

def filter_by_distance(kp, d):
    '''Filter keypoints by selecting the better among neighbors
    
    kp - an array of keypoints
    
    d - the minimum allowed distance
    
    returns a boolean vector of accepted / rejected key points
    '''
    pt = np.array([k.pt for k in kp])
    score = np.array([k.response for k in kp])
    dists = np.sqrt(np.sum((pt[:, np.newaxis, :] - pt[np.newaxis, :, :]) ** 2, 2))
    i, j = np.mgrid[:len(kp), :len(kp)]
    reject = (dists <= d) & ((score[i] < score[j]) | ((score[i] == score[j]) & (i > j)))
    return np.sum(reject, 1) == 0

if __name__ == '__main__':
    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    out_path = sys.argv[3]
    
    img1 = matplotlib.image.imread(img1_path)
    if img1.dtype.itemsize == 2:
        scale1 = 256
    else:
        scale1 = 1
    if img1.ndim == 3:
        img1 = np.mean(img1, 2)
        
    img2 = matplotlib.image.imread(img2_path)
    if img2.dtype.itemsize == 2:
        scale2 = 256
    else:
        scale2 = 1
    if img2.ndim == 3:
        img2 = np.mean(img2, 2)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = min(h1, h2)
    w = min(w1, w2)
    img1 = img1[:h, :w]
    img2 = img2[:h, :w]
        
    sift = cv2.SIFT(sigma=SIFT_SIGMA)
    
    mask = np.ones(img1.shape, np.uint8)
    mask[:50, :] = 0
    mask[-50:, :] = 0
    mask[:, :50] = 0
    mask[:, -50:] = 0
    kp1, desc1 = sift.detectAndCompute((img1/scale1).astype(np.uint8), mask)
    kp2, desc2 = sift.detectAndCompute((img2/scale2).astype(np.uint8), mask)
    f1 = filter_by_distance(kp1, MIN_KP_DISTANCE)
    f2 = filter_by_distance(kp2, MIN_KP_DISTANCE)
    kp1, kp2 = \
      [[x for x, f in zip(y, ff) if f] for y, ff in ((kp1, f1), (kp2, f2))]
    desc1 = desc1[f1]
    desc2 = desc2[f2]
    
    matches = sorted(cv2.BFMatcher().match(desc1, desc2),
                     key=lambda x:x.distance)
    matches = matches[:N_MATCHES]
    idx1 = np.array([m.queryIdx for m in matches])
    idx2 = np.array([m.trainIdx for m in matches])
    pt1 = np.array([(kp1[i].pt[1], kp1[i].pt[0]) for i in idx1])
    pt2 = np.array([(kp2[i].pt[1], kp2[i].pt[0]) for i in idx2])
    
    hash = hashlib.md5()
    hash.update(img1)
    hash.update(img2)
    r = np.random.RandomState(np.fromstring(hash.digest(), int))
    
    for i in range(5):
        votes = np.zeros(len(pt1), int)
        for j in range(N_GROUPS):
            for k in range(10): 
                idxs = r.permutation(len(pt1))[:N_PER_GROUP]
                off, rot = compute_alignment(pt1[idxs], pt2[idxs])
                if det(rot) > 0:
                    # if the determinant of the rotation matrix is -1
                    # then it's a reflection and is bogus
                    break
            else:
                # Five times in a row, got a reflection
                continue
            pt2a = np.dot(pt1, rot) - off
            gets_vote = np.sqrt(np.sum((pt2-pt2a)**2, 1)) < ERROR_EPS
            votes[gets_vote] += 1
            
        good_idx = votes >= GOOD_VOTE_COUNT
        if np.sum(good_idx) > 10:
            pt1 = pt1[good_idx]
            pt2 = pt2[good_idx]
            break
        if np.all(votes == 0):
            raise RuntimeError("No concordance found")
        pt1 = pt1[votes > 0]
        pt2 = pt2[votes > 0]
    else:
        raise RuntimeError("Not enough votes to continue: %d" % np.sum(good_idx))
    off, rot = compute_alignment(pt1, pt2)
    
    i, j = np.mgrid[0:h, 0:w]
    i_out = i * rot[0, 0] + j * rot[1, 0] - off[0]
    j_out = i * rot[0, 1] + j * rot[1, 1] - off[1]
    img_out = map_coordinates(img2.astype(float), (i_out, j_out))
    matplotlib.image.imsave(out_path, img_out.astype(img2.dtype), cmap=matplotlib.cm.gray)
    img = np.zeros((h, w*2), np.uint8)
    img[:,:w] = img1 / scale1
    img[:,w:] = img2 / scale2
    figure = matplotlib.figure.Figure()
    canvas = matplotlib.backends.backend_pdf.FigureCanvasPdf(figure) 
    ax = figure.add_axes([.05, .55, .9, .4])
    ax.imshow(img, matplotlib.cm.gray)
    for i in range(len(pt1)):
        ax.plot((pt1[i, 1], pt2[i,1]+w), (pt1[i, 0], pt2[i, 0]))

    ax = figure.add_axes([.05, .05, .4, .4])
    img = np.zeros((h, w, 3))
    img[:, :, 0] = img1.astype(float) / np.max(img1[mask > 0])
    img[:, :, 1] = img_out.astype(float) / np.max(img_out[mask > 0])
    ax.imshow(img)
    
    ax = figure.add_axes([.55, .05, .4, .4])
    ax.set_axis_off()
    table = [["",""] for _ in range(4)]
    for i in range(2):
        for j in range(2):
            table[i][j] = "%.4f" % rot[i,j]
    table[-1][0] = "%.1f" % off[1]
    table[-1][1] = "%.1f" % off[0] 
    ax.table(cellText = table, loc="center")

    figure.savefig(os.path.splitext(out_path)[0] + ".pdf")
    