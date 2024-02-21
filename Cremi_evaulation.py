import numpy as np
import os
import glob
import imageio.v2 as imageio
import pandas as pd
from skimage.segmentation import find_boundaries
import sys
import torch_em
#from skimage.io import imread
from elf.evaluation import dice_score

#(if your ground truth is not binarised, make sure to put the parameter threshold_gt = 0.)


# def dice_score(segmentation, groundtruth, threshold_seg=None, threshold_gt=None):
#     """ Compute the dice score between binarized segmentation and ground-truth.
#     Arguments:
#         segmentation [np.ndarray] - candidate segmentation to evaluate
#         groundtruth [np.ndarray] - groundtruth
#         threshold_seg [float] - the threshold applied to the segmentation.
#             If None the segmentation is not thresholded.
#         threshold_gt [float] - the threshold applied to the ground-truth.
#             If None the ground-truth is not thresholded.
#     Returns:
#         float - the dice score
#     """
#     assert segmentation.shape == groundtruth.shape, f"{segmentation.shape}, {groundtruth.shape}"
#     if threshold_seg is None:
#         seg = segmentation
#     else:
#         seg = segmentation > threshold_seg
#     if threshold_gt is None:
#         gt = groundtruth
#     else:
#         gt = groundtruth > threshold_gt

#     nom = 2 * np.sum(gt * seg)
#     denom = np.sum(gt) + np.sum(seg)

#     eps = 1e-7
#     score = float(nom) / float(denom + eps)
#     return score

cell_types = ["sampleA-", "sampleB-","sampleC-"]


bd_eval_scores = np.zeros(3)
bd_final_list = []



for ind,i in enumerate(cell_types):


    bd_dir = os.path.join("/scratch-grete/usr/nimmahen/models/Unet/prediction/new_cremi_allpersample/boundaries/", i+"*")
    n = 0 
    bd_dsl = []   
    bd_ds_dict = {}
    for bd_pred_seg in glob.glob(bd_dir):
        filename = os.path.split(bd_pred_seg)[-1]
        gt_path = os.path.join("/scratch-grete/usr/nimmahen/data/Cremi/test_label/", filename)
        

        bd_pred_y = imageio.imread(bd_pred_seg)
        bd_pred_y = np.squeeze(bd_pred_y, axis=0)
        #bd_pred_y = np.where(bd_pred_y>.5,1,0) #changed to binarize
        gt_y = imageio.imread(gt_path)
        bd_gt = find_boundaries(gt_y)
        bd_ds = dice_score(bd_pred_y, bd_gt, threshold_gt=None, threshold_seg=None)
       
        bd_dsl.append(bd_ds)
        bd_ds_dict.update({filename: bd_ds})
        
    bd_dsl = sum(bd_dsl)/len(bd_dsl)
    bd_eval_scores[ind] = bd_dsl
    bd_final_list.append(bd_ds_dict)




#top_5_keys = sorted(final_list[0], key=lambda x: final_list[0][x], reverse=True)[:5]
bd_all_values = [value for dictionary in bd_final_list for value in dictionary.values()]

# Sort the values and get the top 5
bd_top_5_values = sorted(bd_all_values, reverse=True)[:5]

# Get the keys corresponding to the top 5 values
bd_top_5_keys = [key for dictionary in bd_final_list for key, value in dictionary.items() if value in bd_top_5_values]

with open(r'/home/nimmahen/code/results/Unet_new_cremi_allpersample_bd_samples.txt', 'w') as fp:
    for item in bd_top_5_keys:
        # write each item on a new line
        fp.write("%s\n" % item)

with open(r'/home/nimmahen/code/results/Unet_new_cremi_allpersample_bd_scores.txt', 'w') as fp:
    for item in bd_eval_scores:
        # write each item on a new line
        fp.write("%s\n" % item)

print('Unet_new_cremi_allpersample_bd_scores',bd_eval_scores)
#print('dict of images', final_list)










# print(dsl)
# print("avg dice score", sum(dsl)/len(dsl))
# import matplotlib.pyplot as plt
# n = 5 
# fig, ax = plt.subplots(5, n, figsize=(30, 15))


# for i, id in enumerate(top_5_keys):
#     cell_type = id.split("_")[0]
    
#     seg_unetr = os.path.join("/scratch/users/menayat/models/cremi-unetr/predictions/boundaries/", id)
#     seg_swinunetr = os.path.join("/scratch/users/menayat/models/cremi-swin/predictions/boundaries/", id)
#     seg_unet = os.path.join("/scratch/users/menayat/models/cremi-unet/predictions/boundaries/", id)
#     gt_pth = os.path.join("/scratch/users/menayat/data/cremi/test_label/", id)# changed from corrected file
#     img_pth = os.path.join("/scratch/users/menayat/data/cremi/test_image/",id)  ### changed from test/val images

#     pred_unet = imread(seg_unet)
#     pred_swinunetr = imread(seg_swinunetr)
#     pred_unetr = imread(seg_unetr)
#     gt = imread(gt_pth)
#     #gt = find_boundaries(gt_y,mode='thick').astype(np.uint8)
#     raw = imread(img_pth)
#     #breakpoint()
    
#     # if i == n//2:
#     #     ax[0][i].set_title("raw data",fontsize=20,loc='center')
#     #     ax[1][i].set_title("unetr prediction",fontsize=20,loc='center')
#     #     ax[2][i].set_title("swin prediction",fontsize=20,loc='center')
#     #     ax[3][i].set_title("unet prediction",fontsize=20,loc='center')
#     #     ax[4][i].set_title("ground truth",fontsize=20,loc='center')
#     ax[0][i].imshow( raw, cmap='Blues')
#     ax[1][i].imshow(pred_unetr.squeeze())
#     ax[2][i].imshow(pred_swinunetr.squeeze())
#     ax[3][i].imshow(pred_unet.squeeze())
#     ax[4][i].imshow(gt.squeeze())
#     #ax[i].imshow(img_new)\n",

# cell_name =[]
# for i in top_5_keys:
#         cell = i.split(".")[0]
#         cell_name.append(cell)
# for axs, cell in zip(ax[0], cell_name):
#     axs.set_title(f'{cell}', size=18)

# for ax, model in zip(ax[:, 0], ['Raw', 'UNETR', 'SWIN', 'UNET', 'Ground Truth']):
#     ax.set_ylabel(model, size=18)

# # axs[0, 1].annotate('Segmentation trained on A172', (0.5, 1), xytext=(0, 30),
# #                    textcoords='offset points', xycoords='axes fraction',
# #                    ha='center', va='bottom', size=20)
# plt.show()
# fig.savefig('/usr/users/menayat/code-torch-em/UNETR_cremi.png')
    


            

