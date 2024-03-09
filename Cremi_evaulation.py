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
import matplotlib.pyplot as plt



np.random.seed(42)

def generate_random_colormap(num_colors):
    np.random.seed(42)

    colors = np.random.rand(num_colors, 3)
    return ListedColormap(colors)

# def generate_random_colormap(num_colors, background_index=0):
#     np.random.seed(42)

#     # Generate random colors excluding the background color for index 0
#     colors = np.random.rand(num_colors - 1, 3)

#     # Insert a placeholder color for the background index
#     colors = np.insert(colors, background_index, [0, 0, 0], axis=0)

#     return ListedColormap(colors)

cell_types = ["sampleA-", "sampleB-","sampleC-"]


bd_eval_scores = np.zeros(3)
bd_final_list = []

ins_eval_scores = np.zeros(3)
ins_final_list = []



for ind,i in enumerate(cell_types):


    bd_dir = os.path.join("/scratch-grete/usr/nimmahen/models/UNETR/sam/prediction/last_cremi_allpersample_vit_l/boundaries/", i+"*")
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
        bd_ds = round(float(bd_ds), 3)
        bd_dsl.append(bd_ds)
        bd_ds_dict.update({filename: bd_ds})
        
    bd_dsl = sum(bd_dsl)/len(bd_dsl)
    bd_dsl = round(float(bd_dsl), 3)
    bd_eval_scores[ind] = bd_dsl
    bd_final_list.append(bd_ds_dict)


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




bd_all_values = [value for dictionary in bd_final_list for value in dictionary.values()]

# Sort the values and get the top 5
bd_top_5_values = sorted(bd_all_values, reverse=True)[:3]

# Get the keys corresponding to the top 5 values along with their values
bd_top_5_items = [{ key, value} for dictionary in bd_final_list for key, value in dictionary.items() if value in bd_top_5_values]
bd_top_5_keys = [key for dictionary in bd_final_list for key, value in dictionary.items() if value in bd_top_5_values]


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

###### boundary
n = 6
fig, ax = plt.subplots(len(bd_top_5_keys), n, figsize=(45, 20), gridspec_kw={'wspace': 0.1, 'hspace': 0.1})

if len(ins_top_5_keys) == 1:
    ax = np.atleast_2d(ax)



for i, id in enumerate(bd_top_5_keys):
    cell_type = id.split("_")[0]
    
    UNETR_sam_vit_l = os.path.join("/scratch-grete/usr/nimmahen/models/UNETR/sam/prediction/last_cremi_allpersample_vit_l/boundaries/", id)
    UNETR_sam_vit_b = os.path.join("/scratch-grete/usr/nimmahen/models/UNETR/sam/prediction/last_cremi_allpersample_vit_b/boundaries/", id)
    UNETR_sc = os.path.join("/scratch-grete/usr/nimmahen/models/UNETR/sc/prediction/last_cremi_allpersample/boundaries/", id)
    UNET = os.path.join("/scratch-grete/usr/nimmahen/models/Unet/prediction/last_cremi_allpersample/boundaries/", id)

    gt_pth = os.path.join("/scratch-grete/usr/nimmahen/data/Cremi/test_label/", id)
    img_pth = os.path.join("/scratch-grete/usr/nimmahen/data/Cremi/test_image/",id)

    pred_UNETR_sam_vit_l = imageio.imread(UNETR_sam_vit_l)
    pred_UNETR_sam_vit_b = imageio.imread(UNETR_sam_vit_b)
    pred_UNETR_sc = imageio.imread(UNETR_sc)
    pred_UNET = imageio.imread(UNET)
    gt = imageio.imread(gt_pth)
    bd_gt = find_boundaries(gt)
    raw = imageio.imread(img_pth)

   

   
    # import numpy as np

    

    # # Assuming pred_UNETR_sam_vit_l, pred_UNETR_sam_vit_b, pred_UNETR_sc, pred_UNET are your model predictions
    # #predictions = [pred_UNETR_sam_vit_l, pred_UNETR_sam_vit_b, pred_UNETR_sc, pred_UNET]
    # #predictions = [pred_UNET, pred_UNETR_sam_vit_l, pred_UNETR_sam_vit_b ]
    # predictions = [pred_UNETR_sam_vit_l, pred_UNET]


    # # Assign unique colors to each model
    # # colors = ['red', 'green', 'blue', 'purple']
    # # color_vectors = [(255,0,0), (0,255,0), (0,0,255)]

    # colors = ['red', 'green']
    # color_vectors = [(255,0,0), (0,255,0)]

    # # Determine the shape of the composite image based on the first prediction
    # composite_shape = predictions[0].shape
    

    # # Create an empty array to store the composite image
    # #composite_image = np.zeros( composite_shape + (3,))
    # composite_image = np.zeros( composite_shape + (3,))
    
    

    # # Combine predictions with different colors
    # for i, prediction in enumerate(predictions):
    #     #prediction = np.squeeze(prediction)
    #     mask = prediction >  0.5
    #     #composite_image[mask] += np.array(plt.cm.colors.to_rgba(colors[i])[:3]) * 255
    #     composite_image[mask] += np.array(color_vectors[i])

    
    # composite_image = np.squeeze(composite_image)
    

    # # Display the composite image
    # plt.imshow(composite_image)
    # plt.show()

    # fig.savefig('/home/nimmahen/code/Figures/cremi_30persample_boundary_best-worst.png')
    # breakpoint()


 
    

    # ax[0][i].imshow( raw, cmap='gray')
    # ax[1][i].imshow(pred_UNETR_sam_vit_l.squeeze(), cmap=pred_UNETR_sam_vit_l_random_cmap)
    # ax[2][i].imshow(pred_UNETR_sam_vit_b.squeeze(), cmap=pred_UNETR_sam_vit_b_random_cmap)
    # ax[3][i].imshow(pred_UNETR_sc.squeeze(), cmap=pred_UNETR_sc_random_cmap)
    # ax[4][i].imshow(pred_UNET.squeeze(),cmap=pred_UNET_random_cmap)
    # ax[5][i].imshow(gt.squeeze(), cmap='viridis')

    #ax[i].imshow(img_new)\n",
    ax[i][0].imshow(raw, cmap='gray')
    ax[i][1].imshow(pred_UNETR_sam_vit_l.squeeze(), cmap='viridis')
    ax[i][2].imshow(pred_UNETR_sam_vit_b.squeeze(), cmap='plasma')
    ax[i][3].imshow(pred_UNETR_sc.squeeze(), cmap='viridis')
    ax[i][4].imshow(pred_UNET.squeeze(), cmap='plasma')
    ax[i][5].imshow(bd_gt.squeeze(), cmap='viridis')

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
fig.savefig('/home/nimmahen/code/Figures/cremi_allpersample_boundary.pdf', dpi=300)
fig.savefig('/home/nimmahen/code/Figures/cremi_allpersample_boundary.png', dpi=300)
fig.savefig('/home/nimmahen/code/Figures/cremi_allpersample_boundary.svg', dpi=300)







#######instance

n = 6
fig, ax = plt.subplots(len(ins_top_5_keys), n, figsize=(45, 20), gridspec_kw={'wspace': 0.1, 'hspace': 0.1})

if len(ins_top_5_keys) == 1:
    ax = np.atleast_2d(ax)



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
    
    gt_num_classes = len(np.unique(gt))
    gt_random_cmap = generate_random_colormap(gt_num_classes)

    # ax[0][i].imshow( raw, cmap='gray')
    # ax[1][i].imshow(pred_UNETR_sam_vit_l.squeeze(), cmap=pred_UNETR_sam_vit_l_random_cmap)
    # ax[2][i].imshow(pred_UNETR_sam_vit_b.squeeze(), cmap=pred_UNETR_sam_vit_b_random_cmap)
    # ax[3][i].imshow(pred_UNETR_sc.squeeze(), cmap=pred_UNETR_sc_random_cmap)
    # ax[4][i].imshow(pred_UNET.squeeze(),cmap=pred_UNET_random_cmap)
    # ax[5][i].imshow(gt.squeeze(), cmap='viridis')

    #ax[i].imshow(img_new)\n",
    ax[i][0].imshow(raw, cmap='gray')
    ax[i][1].imshow(pred_UNETR_sam_vit_l.squeeze(), interpolation = 'nearest', cmap=pred_UNETR_sam_vit_l_random_cmap)
    ax[i][2].imshow(pred_UNETR_sam_vit_b.squeeze(), interpolation = 'nearest', cmap=pred_UNETR_sam_vit_b_random_cmap)
    ax[i][3].imshow(pred_UNETR_sc.squeeze(), interpolation = 'nearest', cmap=pred_UNETR_sc_random_cmap)
    ax[i][4].imshow(pred_UNET.squeeze(), interpolation = 'nearest', cmap=pred_UNET_random_cmap)
    ax[i][5].imshow(gt.squeeze(), interpolation = 'nearest', cmap=gt_random_cmap)

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
fig.savefig('/home/nimmahen/code/Figures/cremi_allpersample_instance.pdf', dpi=300)
fig.savefig('/home/nimmahen/code/Figures/cremi_allpersample_instance.png', dpi=300)
fig.savefig('/home/nimmahen/code/Figures/cremi_allpersample_instance.svg', dpi=300)


    


            


