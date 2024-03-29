# import matplotlib.pyplot as plt

# # Your list of lists with float numbers
# data_UNETR_sam_new__cremi_vitb = [
#     [0.5710718822010294, 0.5049826725273803,0.5557044297494914],
#     [0.6470019267694128, 0.603416675226591,0.6427943403732681],
#     [0.6556531217330349, 0.6135453340332674, 0.6606195887851634],
#     [0.6857622313922611, 0.6688244848270679, 0.7051000850066987],
#     [0.7033480622136743, 0.707165687841337, 0.7543637555583697]
# ]

# # Flatten the data to a single list
# flattened_data = [item for sublist in data_UNETR_sam_new__cremi_vitb for item in sublist]

# # Plot the flattened data
# plt.plot(flattened_data, marker='o')

# # Add labels and title
# plt.xlabel('number of samples')
# plt.ylabel('dice score')
# plt.title('UNETR_sam_new__cremi_vitb_dicescore')

# # Show the plot
# plt.show()
# plt.savefig('/home/nimmahen/code/plots/UNETR_sam_new__cremi_vitb_dicescore.png')

###################################

# import matplotlib.pyplot as plt
# import numpy as np

# # Your list of lists with float numbers and group information
# data = [
#     {'group': '10sample', 'values': [0.5710718822010294, 0.5049826725273803,0.5557044297494914]},
#     {'group': '30sample', 'values': [0.6470019267694128, 0.603416675226591,0.6427943403732681]},
#     {'group': '50sample', 'values': [0.6556531217330349, 0.6135453340332674, 0.6606195887851634]},
#     {'group': '75sample', 'values': [0.6857622313922611, 0.6688244848270679, 0.7051000850066987]},
#     {'group': 'all sample', 'values': [0.7033480622136743, 0.707165687841337, 0.7543637555583697]}
# ]

# # Flatten the data and create corresponding x-axis values for each group
# x_values = np.arange(len(data[0]['values']))
# for group_data in data:
#     plt.plot(x_values, group_data['values'], marker='o', label=f"Group {group_data['group']}")

# # Add labels and title
# plt.xlabel('number of samples')
# plt.ylabel('dice scores')
# plt.title('UNETR_sam_new__cremi_vitb_dicescore')
# plt.legend()

# # Show the plot
# plt.show()
# plt.savefig('/home/nimmahen/code/plots/UNETR_sam_new__cremi_vitb_dicescore_grouped.png')


###########################################
# import matplotlib.pyplot as plt
# import numpy as np

# # Your list of dictionaries with float numbers and group information
# data = [
#     {'group': '10sample', 'values': [0.5710718822010294, 0.5049826725273803, 0.5557044297494914]},
#     {'group': '30sample', 'values': [0.6470019267694128, 0.603416675226591, 0.6427943403732681]},
#     {'group': '50sample', 'values': [0.6556531217330349, 0.6135453340332674, 0.6606195887851634]},
#     {'group': '75sample', 'values': [0.6857622313922611, 0.6688244848270679, 0.7051000850066987]},
#     {'group': 'all sample', 'values': [0.7033480622136743, 0.707165687841337, 0.7543637555583697]}
# ]

# # Flatten the data and create clustered x-axis values
# x_values = np.arange(len(data[0]['values'])) + np.tile(np.arange(len(data)), len(data[0]['values']))

# # Plot the clustered values
# for i, group_data in enumerate(data):
#     plt.plot(x_values[i * len(group_data['values']): (i + 1) * len(group_data['values'])],
#              group_data['values'], marker='o', label=f"{group_data['group']}")

# # Add labels and title
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.title('Clustered Plot of Lists')
# plt.legend()

# # Show the plot
# plt.show()

# plt.savefig('/home/nimmahen/code/plots/UNETR_sam_new__cremi_vitb_dicescore_clustered.png')


#####################################

# plt.figure(figsize=(12,4))
# plt.subplot(1,2,1)
# plt.plot(train_losses, label='Train')
# plt.plot(val_losses, label='Val')
# plt.title('Training and validation losses')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()

# plt.subplot(1,2,2)
# plt.plot(val_accuracies, label='acc')
# plt.title('val accuracies')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.show()
# plt.savefig('/home/nimmahen/code/practice/flowers_train_val_curves.png')


################

import matplotlib.pyplot as plt
import numpy as np

# UNETR_sam_vit_b_Cremi = [0.543, 0.630, 0.642, 0.686, 0.721]
# UNETR_sc_Cremi = [0.573, 0.597, 0.627, 0.637, 0.701]
# UNET_Cremi = [0.474, 0.444, 0.406, 0.475, 0.370]
# sample_numbers = ['10', '30', '50', '75', 'all']

###########################################################################################################
# ## LiveCell MCF7 foreground dice score
# UNETR_sc = [0.730, 0.770, 0.767, 0.788]
# UNETR_sam_vit_b = [0.752, 0.789, 0.795, 0.815]
# UNET = [0.772, 0.840, 0.868, 0.876]
# sample_numbers = ['10', '50', '100', 'all']
########
## LiveCell MCF7 boundary dice score
# UNETR_sc = [0.326, 0.368, 0.373, 0.394]
# UNETR_sam_vit_b = [0.348, 0.402, 0.409, 0.432]
# UNET = [0.358, 0.444, 0.486, 0.501]
# sample_numbers = ['10', '50', '100', 'all']
#########

# # ## LiveCell MCF7 instance  msa
# UNETR_sc = [0.049, 0.076, 0.085, 0.098]
# UNETR_sam_vit_b = [0.10, 0.148, 0.162, 0.212]
# UNET = [0.111, 0.139, 0.169, 0.198]
# sample_numbers = ['10', '50', '100', 'all']
##################################################################################################

# #######
# ### LiveCell all foreground dice score
# UNETR_sc = [0.852, 0.769, 0.580, 0.723, 0.815, 0.728, 0.874, 0.847]
# UNETR_sam_vit_b = [0.881, 0.865, 0.699, 0.842, 0.872, 0.804, 0.916, 0.881]
# UNETR_sam_vit_l = [0.870, 0.838, 0.637, 0.794, 0.843, 0.768, 0.904, 0.863]
# UNET = [ 0.941, 0.882,  0.854, 0.906, 0.894, 0.836 ,0.930, 0.944]
# sample_numbers = ["A172", "BT474", "BV2", "Huh7", "MCF7", "SHSY5Y", "SkBr3", "SKOV3"]
# #####

# ### LiveCell all boundary dice score
# UNETR_sc = [0.390, 0.362, 0.388, 0.212, 0.435, 0.467, 0.523, 0.376]
# UNETR_sam_vit_b = [0.433, 0.441, 0.447, 0.302, 0.494, 0.504, 0.587, 0.418]
# UNETR_sam_vit_l = [0.411, 0.367, 0.382, 0.198, 0.449, 0.491, 0.534, 0.404]
# UNET = [ 0.469 ,0.501, 0.624, 0.425, 0.532, 0.555, 0.618, 0.451]
# sample_numbers = ["A172", "BT474", "BV2", "Huh7", "MCF7", "SHSY5Y", "SkBr3", "SKOV3"]
# ##########

# ### LiveCell all instance msa
# UNETR_sc = [0.138, 0.117, 0.320, 0.081, 0.127, 0.095, 0.522, 0.182]
# UNETR_sam_vit_b = [0.200, 0.191, 0.462, 0.196, 0.214, 0.136, 0.612, 0.261]
# UNETR_sam_vit_l = [0.255, 0.259, 0.428, 0.337, 0.283, 0.154, 0.632, 0.307]
# UNET = [ 0.194, 0.216, 0.497, 0.277, 0.214, 0.124, 0.565, 0.228]
# sample_numbers = ["A172", "BT474", "BV2", "Huh7", "MCF7", "SHSY5Y", "SkBr3", "SKOV3"]
# ##########
####################################################################################################

### LiveCell all 60sample foreground dice score
UNETR_sc = [0.855, 0.778, 0.556, 0.740, 0.812, 0.738, 0.878, 0.852]
UNETR_sam_vit_b = [0.871, 0.843, 0.671, 0.811, 0.854, 0.780, 0.904, 0.873]
UNETR_sam_vit_l = [0.861, 0.825, 0.611, 0.771, 0.832, 0.759, 0.893, 0.855]
UNET = [0.941, 0.882, 0.854, 0.906, 0.894, 0.836, 0.930, 0.944]
sample_numbers = ["A172", "BT474", "BV2", "Huh7", "MCF7", "SHSY5Y", "SkBr3", "SKOV3"]
#####

# ### LiveCell all 60sample boundary dice score
# UNETR_sc = [0.382, 0.341, 0.374, 0.199, 0.413, 0.458, 0.517, 0.368]
# UNETR_sam_vit_b = [0.416, 0.401, 0.413, 0.257, 0.462, 0.466, 0.561, 0.403]
# UNETR_sam_vit_l = [0.416, 0.389, 0.378, 0.214, 0.46, 0.486, 0.537, 0.405]
# UNET = [0.469, 0.501, 0.624, 0.425, 0.532, 0.555, 0.618, 0.451]
# sample_numbers = ["A172", "BT474", "BV2", "Huh7", "MCF7", "SHSY5Y", "SkBr3", "SKOV3"]
##########

# ### LiveCell all 60sample instance msa
# UNETR_sc = [0.120, 0.092, 0.293, 0.073, 0.096, 0.081, 0.516, 0.163]
# UNETR_sam_vit_b = [0.173, 0.141, 0.368, 0.169, 0.153, 0.090, 0.556, 0.253]
# UNETR_sam_vit_l = [0.259, 0.262, 0.425, 0.333, 0.288, 0.147, 0.626, 0.302]
# UNET = [0.194, 0.216, 0.497, 0.277, 0.214, 0.124, 0.565, 0.228]
# sample_numbers = ["A172", "BT474", "BV2", "Huh7", "MCF7", "SHSY5Y", "SkBr3", "SKOV3"]
##########


plt.figure(figsize=(15,5))
plt.plot(sample_numbers, UNETR_sc, marker='o', linestyle='-', label='UNETR_sc')
plt.plot(sample_numbers, UNETR_sam_vit_b, marker='o', linestyle='-', label='UNETR_sam_vit_b')
plt.plot(sample_numbers, UNETR_sam_vit_l, marker='o', linestyle='-', label='UNETR_sam_vit_l')
plt.plot(sample_numbers, UNET, marker='o', linestyle='-', label='UNET')
plt.title('LiveCell all 60sample foreground dice score')
plt.xlabel('cell type') #number of samples
plt.ylabel('dice score') #dice score/ Mean Segmentation Accuracy
plt.xticks(sample_numbers, labels=sample_numbers)
for i, y in enumerate(UNETR_sc):
    plt.text(i, y, str(y), ha='center', va='bottom')
for i, y in enumerate(UNETR_sam_vit_b):
    plt.text(i, y, str(y), ha='center', va='bottom')
for i, y in enumerate(UNETR_sam_vit_l):
    plt.text(i, y, str(y), ha='center', va='bottom')
for i, y in enumerate(UNET):
    plt.text(i, y, str(y), ha='center', va='bottom')

plt.legend()



plt.show()
plt.savefig('/home/nimmahen/code/plots/last_plots/LiveCell all 60sample foreground dice score 4 models.png')


# first idea to plot :)
# x_foreground = ["UNETR_sc","UNETR_sam_vit_b","UNET"]
# y_foreground = [(0.776,0.774),(0.826,0.845),(0.882,0.898)] 
