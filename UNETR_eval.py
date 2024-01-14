import numpy as np
import os
import glob
import imageio.v2 as imageio
import pandas as pd
from skimage.segmentation import find_boundaries
import sys
import torch_em

# (if your ground truth is not binarised, make sure to put the parameter threshold_gt = 0.


def dice_score(segmentation, groundtruth, threshold_seg=None, threshold_gt=None):
    """ Compute the dice score between binarized segmentation and ground-truth.
    Arguments:
        segmentation [np.ndarray] - candidate segmentation to evaluate
        groundtruth [np.ndarray] - groundtruth
        threshold_seg [float] - the threshold applied to the segmentation.
            If None the segmentation is not thresholded.
        threshold_gt [float] - the threshold applied to the ground-truth.
            If None the ground-truth is not thresholded.
    Returns:
        float - the dice score
    """
    assert segmentation.shape == groundtruth.shape, f"{segmentation.shape}, {groundtruth.shape}"
    if threshold_seg is None:
        seg = segmentation
    else:
        seg = segmentation > threshold_seg
    if threshold_gt is None:
        gt = groundtruth
    else:
        gt = groundtruth > threshold_gt

    nom = 2 * np.sum(gt * seg)
    denom = np.sum(gt) + np.sum(seg)

    eps = 1e-7
    score = float(nom) / float(denom + eps)
    return score

cell_types = ["MCF7"] #"A172", "BT474","BV2", "Huh7", "MCF7", "SHSY5Y", "SkBr3", "SKOV3"
eval_scores = np.zeros(1)
for ind,i in enumerate(cell_types):

    seg_x = os.path.join("/scratch-grete/usr/nimmahen/models/UNETR/sc/prediction/livecell_MCF7_10/boundaries", i+"*")
    n = 0
    dsl = []
    for pred_seg in glob.glob(seg_x):
        filename = os.path.split(pred_seg)[-1]
        gt_path = os.path.join("/scratch-grete/projects/nim00007/data/LiveCELL/annotations_corrected/livecell_test_images/",i,  filename)
        

        pred_y = imageio.imread(pred_seg)
        gt_y = imageio.imread(gt_path)
        gt_y = find_boundaries(gt_y,mode='thick')#.astype(np.uint8)
        
        ds = dice_score(pred_y, gt_y, threshold_gt= 0)
        dsl.append(ds)
    dsl = sum(dsl)/len(dsl)
    eval_scores[ind] = dsl
with open(r'/home/nimmahen/code/results/UNETR_sc_livecell_MCF7_10_vit_b_bddtype.txt', 'w') as fp:
    for item in eval_scores:
        # write each item on a new line
        fp.write("%s\n" % item)
print('UNETR_sc_livecell_MCF7_10_vit_b_bd_dtype',eval_scores)

# print(dsl)
# print("avg dice score", sum(dsl)/len(dsl))

    


            


