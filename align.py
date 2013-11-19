'''
Created on Nov 15, 2013

@author: Lee Kamentsky
'''
import argparse
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
N_GROUPS = 20
MAX_OUTLIER_RUNS = 200
N_GOOD_OUTLIER_RUNS = 5
N_PER_GROUP = 5
ERROR_EPS = 5.0
GOOD_VOTE_COUNT = 15
SIFT_SIGMA = 5
MIN_KP_DISTANCE = 5
assert hasattr(matplotlib.cm, "gray")
ALG_SIFT = "SIFT"
ALG_SURF = "SURF"
ALG_BRIEF = "BRIEF"
ALG_MSER = "MSER"
ALG_ORB = "ORB"
ALG_BRISK = "BRISK"
ALG_FREAK = "FREAK"
ALG_ALL = [ALG_SIFT, ALG_SURF, ALG_BRIEF, ALG_MSER, ALG_ORB, ALG_BRISK, ALG_FREAK]

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

def filter_outliers(pt1, pt2, r):
    '''Filter outliers that never agree with a consensus
    
    pt1, pt2 - N vectors of dimension D giving corresponding keypoints
               in each image.
               
    r - a numpy.random.RandomState to be used for permuting
        the voter choices
               
    returns pt1, pt2 with the outliers removed.
    '''
    votes = np.zeros(len(pt1), int)
    in_voting_group = np.zeros(len(pt1), int)
    n_good_outlier_runs = 0
    n_outlier_runs = 0
    while n_good_outlier_runs < N_GOOD_OUTLIER_RUNS and n_outlier_runs < MAX_OUTLIER_RUNS:
        idxs = r.permutation(len(pt1))[:N_PER_GROUP]
        off, rot = compute_alignment(pt1[idxs], pt2[idxs])
        if det(rot) > 0:
            pt2a = np.dot(pt1, rot) - off
            gets_vote = np.sqrt(np.sum((pt2-pt2a)**2, 1)) < ERROR_EPS
            if np.any(gets_vote):
                in_voting_group[idxs] += 1
                votes[gets_vote] += 1
                n_good_outlier_runs += 1
            n_outlier_runs += 1
    to_keep = (votes > 0) | (in_voting_group > 0)
    return pt1[to_keep], pt2[to_keep]

def filter_by_voting(pt1, pt2, r):
    '''Filter by creating weak voters on concordance
    
    Choose a number of keypoint pairs from the set and compute
    an alignment. Score all keypoint pairs using the alignment
    and register # of votes per pair. Accept all pairs whose
    # of votes exceeds a threshold.
    
    pt1, pt2 - locations of keypoint pairs
    
    r - a numpy.random.RandomState for doing the permutations
    '''
    while len(pt1) > 2*N_PER_GROUP:
        votes = np.zeros(len(pt1), int)
        n_runs = 0
        while n_runs < N_GROUPS:
            idxs = r.permutation(len(pt1))[:N_PER_GROUP]
            off, rot = compute_alignment(pt1[idxs], pt2[idxs])
            if det(rot) > 0:
                pt2a = np.dot(pt1, rot) - off
                gets_vote = np.sqrt(np.sum((pt2-pt2a)**2, 1)) < ERROR_EPS
                votes[gets_vote] += 1
                n_runs += 1
            
        good_idx = votes >= GOOD_VOTE_COUNT
        if np.sum(good_idx) > len(pt1) / 2:
            pt1 = pt1[good_idx]
            pt2 = pt2[good_idx]
            break
        if np.all(votes == 0):
            raise RuntimeError("No concordance found")
        min_vote_total = np.min(votes)
        if min_vote_total < np.max(votes):
            pt1 = pt1[votes > min_vote_total]
            pt2 = pt2[votes > min_vote_total]
        else:
            to_remove = r.randint(len(pt1))
            pt1 = np.delete(pt1, to_remove, 0)
            pt2 = np.delete(pt2, to_remove, 0)
    return pt1, pt2        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Align two images")
    parser.add_argument('reference', action="store", 
                        help = "The reference image")
    parser.add_argument('target', action="store",
                        help = "The image to be aligned to the reference")
    parser.add_argument('output', action="store",
                        help = "The name for the output image")
    parser.add_argument('--scale', action="store",
                        default=5, type=float,
                        help = "The scale of distinguishing features in the image (in pixels)",
                        dest="scale")
    parser.add_argument('--algorithm', action='store', 
                        choices = ALG_ALL, default=ALG_SIFT,
                        help = "The keypoint detection algorithm",
                        dest="keypoint_detector")
    result = parser.parse_args()
    img1_path = result.reference
    img2_path = result.target
    out_path = result.output
    
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
        
    if result.keypoint_detector == ALG_SIFT:
        detector = cv2.SIFT(sigma=result.scale)
        extractor = detector
    elif result.keypoint_detector == ALG_BRISK:
        detector = cv2.BRISK(patternScale=result.scale)
        extractor = detector
    else:
        detector = cv2.FeatureDetector_create(result.keypoint_detector)
        extractor = cv2.DescriptorExtractor_create(result.keypoint_detector)
    mask = np.ones(img1.shape, np.uint8)
    mask[:50, :] = 0
    mask[-50:, :] = 0
    mask[:, :50] = 0
    mask[:, -50:] = 0
    img1u8 = (img1/scale1).astype(np.uint8)
    img2u8 = (img2/scale2).astype(np.uint8)
    kp1 = detector.detect(img1u8, mask)
    kp2 = detector.detect(img2u8, mask)
    f1 = filter_by_distance(kp1, MIN_KP_DISTANCE)
    f2 = filter_by_distance(kp2, MIN_KP_DISTANCE)
    kp1, kp2 = \
      [[x for x, f in zip(y, ff) if f] for y, ff in ((kp1, f1), (kp2, f2))]
    kp1, desc1 = extractor.compute(img1u8, kp1)
    kp2, desc2 = extractor.compute(img2u8, kp2)
    
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
    
    pt1, pt2 = filter_outliers(pt1, pt2, r)
    if len(pt1) == 0:
        raise RuntimeError("No alignment concordance could be found")
    
    pt1, pt2 = filter_by_voting(pt1, pt2, r)
    if len(pt1) == 0:
        raise RuntimeError("No alignment concordance could be found")

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
    