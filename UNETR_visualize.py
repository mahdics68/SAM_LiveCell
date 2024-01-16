import numpy as np
import os
import glob
import imageio.v2 as imageio
import pandas as pd
from skimage.segmentation import find_boundaries
import sys
import torch_em
import matplotlib.pyplot as plt


img_dir = "MCF7_Phase_H4_1_00d00h00m_1.tif"

seg_x = os.path.join("/scratch-grete/usr/nimmahen/models/UNETR/sam/prediction/livecell_MCF7_10_vit_b_Btch2/segmentation/", img_dir)

filename = os.path.split(seg_x)[-1]
gt_path = os.path.join("/scratch-grete/projects/nim00007/data/LiveCELL/annotations_corrected/livecell_test_images/MCF7/", filename)
raw_img = os.path.join("/scratch-grete/projects/nim00007/data/LiveCELL/images/livecell_test_images/", filename)

prediction = imageio.imread(seg_x)
ground_truth = imageio.imread(gt_path)
image = imageio.imread(raw_img)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original Image
axes[0].imshow(image)
axes[0].set_title('Original Image')

# Ground Truth
axes[1].imshow(ground_truth)
axes[1].set_title('Ground Truth')

# Prediction
axes[2].imshow(prediction)
axes[2].set_title('Prediction')

# Display the plot
plt.show()
fig.savefig('/home/nimmahen/code/plots/livecell_MCF7_10_vit_b_Btch2_seg.png')



#/scratch-grete/usr/nimmahen/models/UNETR/sam/checkpoints/livecell_MCF7_10_vit_b_Btch2/logs/unetr-MCF7/events.out.tfevents.1705009732.ggpu202.1163234.0
            
#livecell_MCF7_10_vit_b_Btch2/

#tensorboard --logdir livecell_MCF7_10_vit_b_Btch2/
