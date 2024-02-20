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

#############
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

# ## LiveCell MCF7 instance  msa
UNETR_sc = [0.049, 0.076, 0.085, 0.098]
UNETR_sam_vit_b = [0.10, 0.148, 0.162, 0.212]
UNET = [0.111, 0.139, 0.169, 0.198]
sample_numbers = ['10', '50', '100', 'all']

# #######
### LiveCell all foreground dice score
UNETR_sc = [0.049, 0.076, 0.085, 0.098]
UNETR_sam_vit_b = [0.881, 0.865, 0.699, 0.842, 0.872, 0.804, 0.916, 0.881]
UNET = [ 0.941, 0.882,  0.854, 0.906, 0.894, 0.836 ,0.930, 0.944]
sample_numbers = ['10', '50', '100', 'all']
#####

### LiveCell all boundary dice score
UNETR_sc = [0.049, 0.076, 0.085, 0.098]
UNETR_sam_vit_b = [0.433, 0.441, 0.447, 0.302, 0.494, 0.504, 0.587, 0.418]
UNET = [ 0.469 ,0.501, 0.624, 0.425, 0.532, 0.555, 0.618, 0.451]
sample_numbers = ['10', '50', '100', 'all']
##########

### LiveCell all instance msa
UNETR_sc = [0.049, 0.076, 0.085, 0.098]
UNETR_sam_vit_b = [0.200, 0.191, 0.462, 0.196, 0.214, 0.136, 0.612, 0.261]
UNET = [ 0.194, 0.216, 0.497, 0.277, 0.214, 0.124, 0.565, 0.228]
sample_numbers = ['10', '50', '100', 'all']
##########


plt.figure(figsize=(12,4))
plt.plot(sample_numbers, UNETR_sc, marker='o', linestyle='-', label='UNETR_sc')
plt.plot(sample_numbers, UNETR_sam_vit_b, marker='o', linestyle='-', label='UNETR_sam_vit_b')
plt.plot(sample_numbers, UNET, marker='o', linestyle='-', label='UNET')
plt.title('LiveCell MCF7 instance msa')
plt.xlabel('number of samples')
plt.ylabel('dice score')
plt.xticks(sample_numbers, labels=sample_numbers)
for i, y in enumerate(UNETR_sc):
    plt.text(i, y, str(y), ha='center', va='bottom')
for i, y in enumerate(UNETR_sam_vit_b):
    plt.text(i, y, str(y), ha='center', va='bottom')
for i, y in enumerate(UNET):
    plt.text(i, y, str(y), ha='center', va='bottom')

plt.legend()



plt.show()
plt.savefig('/home/nimmahen/code/plots/LiveCell MCF7 instance msa.png')
