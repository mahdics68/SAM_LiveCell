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
from elf.evaluation import mean_segmentation_accuracy
from matplotlib.colors import ListedColormap


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

def generate_random_colormap(num_colors):
    colors = np.random.rand(num_colors, 3)
    return ListedColormap(colors)

cell_types = ["sampleA-", "sampleB-","sampleC-"]


bd_eval_scores = np.zeros(3)
bd_final_list = []

ins_eval_scores = np.zeros(3)
ins_final_list = []



for ind,i in enumerate(cell_types):


    # bd_dir = os.path.join("/scratch-grete/usr/nimmahen/models/UNETR/sam/prediction/last_cremi_30persample_vit_l/boundaries/", i+"*")
    # n = 0 
    # bd_dsl = []   
    # bd_ds_dict = {}
    # for bd_pred_seg in glob.glob(bd_dir):
    #     filename = os.path.split(bd_pred_seg)[-1]
    #     gt_path = os.path.join("/scratch-grete/usr/nimmahen/data/Cremi/test_label/", filename)
        

    #     bd_pred_y = imageio.imread(bd_pred_seg)
    #     bd_pred_y = np.squeeze(bd_pred_y, axis=0)
    #     #bd_pred_y = np.where(bd_pred_y>.5,1,0) #changed to binarize
    #     gt_y = imageio.imread(gt_path)
    #     bd_gt = find_boundaries(gt_y)
    #     bd_ds = dice_score(bd_pred_y, bd_gt, threshold_gt=None, threshold_seg=None)
    #     bd_ds = round(float(bd_ds), 3)
    #     bd_dsl.append(bd_ds)
    #     bd_ds_dict.update({filename: bd_ds})
        
    # bd_dsl = sum(bd_dsl)/len(bd_dsl)
    # bd_dsl = round(float(bd_dsl), 3)
    # bd_eval_scores[ind] = bd_dsl
    # bd_final_list.append(bd_ds_dict)


    ins_dir = os.path.join("/scratch-grete/usr/nimmahen/models/UNETR/sam/prediction/last_cremi_allpersample_vit_l/instance/", i+"*")
    n = 0
    ins_msal = []   
    ins_msa_dict = {}
    for ins_pred_seg in glob.glob(ins_dir):
        filename = os.path.split(ins_pred_seg)[-1]
        gt_path = os.path.join("/scratch-grete/usr/nimmahen/data/Cremi/test_label/",  filename)
        

        ins_pred_y = imageio.imread(ins_pred_seg)
        #pred_y = np.where(pred_y>.5,1,0)
        gt_y = imageio.imread(gt_path)
        #gt_y = np.where(gt_y>.5,1,0)
        ins_msa = mean_segmentation_accuracy(ins_pred_y, gt_y)
        ins_msa = round(float(ins_msa), 3)
        ins_msal.append(ins_msa)
        ins_msa_dict.update({filename: ins_msa})
    ins_msal = sum(ins_msal)/len(ins_msal)
    ins_msal = round(float(ins_msal), 3)
    ins_eval_scores[ind] = ins_msal
    ins_final_list.append(ins_msa_dict)




# bd_all_values = [value for dictionary in bd_final_list for value in dictionary.values()]

# # Sort the values and get the top 5
# bd_top_5_values = sorted(bd_all_values, reverse=True)[:5]

# # Get the keys corresponding to the top 5 values along with their values
# bd_top_5_items = [{ key, value} for dictionary in bd_final_list for key, value in dictionary.items() if value in bd_top_5_values]

# # Save the top 5 items with values as a list of dictionaries in a file, separated by commas
# with open('/home/nimmahen/code/results/last_results/UNETR_sam_last_cremi_30persample_vit_l_boundaries_items.txt', 'w') as fp:
#     # write all items in a single line, separated by commas
#     fp.write(', '.join(map(str, bd_top_5_items)))


# bd_eval_scores = bd_eval_scores.tolist()

# with open('/home/nimmahen/code/results/last_results/UNETR_sam_last_cremi_30persample_vit_l_boundaries_scores.txt', 'w') as file:
#     file.write(str(bd_eval_scores))

# print('UNETR_sam_last_cremi_30persample_vit_l_boundaries_scores',bd_eval_scores)
# print(round((sum(bd_eval_scores)/len(bd_eval_scores)),3))






ins_all_values = [value for dictionary in ins_final_list for value in dictionary.values()]

# Sort the values and get the top 5
ins_top_5_values = sorted(ins_all_values, reverse=True)[:3]

# Get the keys corresponding to the top 5 values along with their values
ins_top_5_items = [{ key, value} for dictionary in ins_final_list for key, value in dictionary.items() if value in ins_top_5_values]
ins_top_5_keys = [key for dictionary in ins_final_list for key, value in dictionary.items() if value in ins_top_5_values]

# # Save the top 5 items with values as a list of dictionaries in a file, separated by commas
# with open('/home/nimmahen/code/results/last_results/UNETR_sam_last_cremi_30persample_vit_l_instance_items.txt', 'w') as fp:
#     # write all items in a single line, separated by commas
#     fp.write(', '.join(map(str, ins_top_5_items)))


# ins_eval_scores = ins_eval_scores.tolist()

# with open('/home/nimmahen/code/results/last_results/UNETR_sam_last_cremi_30persample_vit_l_instance_scores.txt', 'w') as file:
#     file.write(str(ins_eval_scores))

# print('UNETR_sam_last_cremi_30persample_vit_l_instance_scores',ins_eval_scores)
# print(round((sum(ins_eval_scores)/len(ins_eval_scores)),3))










import matplotlib.pyplot as plt
n = 6
#fig, ax = plt.subplots(6, n, figsize=(30, 15))
#fig, ax = plt.subplots(n, len(ins_top_5_keys), figsize=(30, 15), gridspec_kw={'wspace': 0.3, 'hspace': 0.3})
fig, ax = plt.subplots(len(ins_top_5_keys), n, figsize=(30, 15), gridspec_kw={'wspace': 0.3, 'hspace': 0.3})



for i, id in enumerate(ins_top_5_keys):
    cell_type = id.split("_")[0]
    
    UNETR_sam_vit_l = os.path.join("/scratch-grete/usr/nimmahen/models/UNETR/sam/prediction/last_cremi_allpersample_vit_l/instance/", id)
    UNETR_sam_vit_b = os.path.join("/scratch-grete/usr/nimmahen/models/UNETR/sam/prediction/last_cremi_allpersample_vit_b/instance/", id)
    UNETR_sc = os.path.join("/scratch-grete/usr/nimmahen/models/UNETR/sc/prediction/last_cremi_allpersample/instance/", id)
    UNET = os.path.join("/scratch-grete/usr/nimmahen/models/Unet/prediction/last_cremi_allpersample/instance/", id)

    gt_pth = os.path.join("/scratch-grete/usr/nimmahen/data/Cremi/test_label/", id)
    img_pth = os.path.join("/scratch-grete/usr/nimmahen/data/Cremi/test_image/",id)

    pred_UNETR_sam_vit_l = imageio.imread(UNETR_sam_vit_l)
    pred_UNETR_sam_vit_b = imageio.imread(UNETR_sam_vit_b)
    pred_UNETR_sc = imageio.imread(UNETR_sc)
    pred_UNET = imageio.imread(UNET)
    gt = imageio.imread(gt_pth)
    raw = imageio.imread(img_pth)

    pred_UNETR_sam_vit_l_num_classes = len(np.unique(pred_UNETR_sam_vit_l))
    pred_UNETR_sam_vit_l_random_cmap = generate_random_colormap(pred_UNETR_sam_vit_l_num_classes)

    pred_UNETR_sam_vit_b_num_classes = len(np.unique(pred_UNETR_sam_vit_b))
    pred_UNETR_sam_vit_b_random_cmap = generate_random_colormap(pred_UNETR_sam_vit_b_num_classes)

    pred_UNETR_sc_num_classes = len(np.unique(pred_UNETR_sc))
    pred_UNETR_sc_random_cmap = generate_random_colormap(pred_UNETR_sc_num_classes)

    pred_UNET_num_classes = len(np.unique(pred_UNET))
    pred_UNET_random_cmap = generate_random_colormap(pred_UNET_num_classes)
    
    # if i == n//2:
    #     ax[0][i].set_title("raw data",fontsize=20,loc='center')
    #     ax[1][i].set_title("unetr prediction",fontsize=20,loc='center')
    #     ax[2][i].set_title("swin prediction",fontsize=20,loc='center')
    #     ax[3][i].set_title("unet prediction",fontsize=20,loc='center')
    #     ax[4][i].set_title("ground truth",fontsize=20,loc='center')

    # ax[0][i].imshow( raw, cmap='gray')
    # ax[1][i].imshow(pred_UNETR_sam_vit_l.squeeze(), cmap=pred_UNETR_sam_vit_l_random_cmap)
    # ax[2][i].imshow(pred_UNETR_sam_vit_b.squeeze(), cmap=pred_UNETR_sam_vit_b_random_cmap)
    # ax[3][i].imshow(pred_UNETR_sc.squeeze(), cmap=pred_UNETR_sc_random_cmap)
    # ax[4][i].imshow(pred_UNET.squeeze(),cmap=pred_UNET_random_cmap)
    # ax[5][i].imshow(gt.squeeze(), cmap='viridis')

    #ax[i].imshow(img_new)\n",
    ax[i][0].imshow(raw, cmap='gray')
    ax[i][1].imshow(pred_UNETR_sam_vit_l.squeeze(), cmap=pred_UNETR_sam_vit_l_random_cmap)
    ax[i][2].imshow(pred_UNETR_sam_vit_b.squeeze(), cmap=pred_UNETR_sam_vit_b_random_cmap)
    ax[i][3].imshow(pred_UNETR_sc.squeeze(), cmap=pred_UNETR_sc_random_cmap)
    ax[i][4].imshow(pred_UNET.squeeze(), cmap=pred_UNET_random_cmap)
    ax[i][5].imshow(gt.squeeze(), cmap='viridis')

    for j in range(n):
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
            ax[i][j].set_xticklabels([])
            ax[i][j].set_yticklabels([])





model_names = ['Raw', 'UNETR_sam_vit_l', 'UNETR_sam_vit_b', 'UNETR_sc', 'UNET', 'Ground Truth']
for ax, model_name in zip(ax[0], model_names):
    ax.set_title(model_name, size=18)


# for ax, key in zip(ax[:, 0], ins_top_5_keys):
#     ax.set_ylabel(key.split(".")[0], size=18)


plt.show()
fig.savefig('/home/nimmahen/code/Figures/cremi_allpersample_instance.png')
    


            


