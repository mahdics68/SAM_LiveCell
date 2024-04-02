import matplotlib.pyplot as plt
import numpy as np

# Your data
all_models = ["UNETR_sc", "UNETR_vit_b", "UNETR_vit_l", "UNET"]

################## LiveCell
#### foreground dice scores
eval_scores = [(0.776, 0.774), (0.826, 0.845), (0.801, 0.815), (0.882, 0.898)]

#### boundaries dice scores
#eval_scores = [(0.382, 0.394), (0.422, 0.453), (0.411, 0.404), (0.498, 0.522)]

### instance msa
#eval_scores = [(0.179, 0.198), (0.238, 0.284), (0.330, 0.332), (0.240, 0.289)]





# Extracting the data for plotting

data = np.array(eval_scores)
bar_width = 0.4
bar_positions = np.arange(len(all_models))

# Plotting
fig, ax = plt.subplots()

legend_labels = ["60 Samples per Cell types", "All Samples"]  # Custom labels for the legend

for i in range(len(data[0])):
    bars = ax.bar(bar_positions - bar_width / 2 + i * bar_width, data[:, i], bar_width, label=legend_labels[i])

    # Adding y-values on the bars
    for bar, y_value in zip(bars, data[:, i]):
        ax.text(bar.get_x() + bar.get_width() / 2, y_value, f'{y_value:.3f}', ha='center', va='bottom')

# Adding labels and title
ax.set_xticks(bar_positions)
ax.set_xticklabels(all_models, fontsize=20)
ax.set_xlabel('Models', fontsize=20)
ax.set_ylabel('Dice Score', fontsize=20)  # Dice Score / Mean Segmentation Accuracy
ax.set_title('Foreground Segmentation Evaluation for LiveCell', fontsize=20)

# Adding custom legend
ax.legend(loc='lower right')

# Show the plot
plt.show()


plt.savefig('/home/nimmahen/code/plots/last_plots/LiveCell all average foreground dice score 4 models.png')


#####MCF7 livecell

# #### foreground dice score 
# eval_scores = [0.788, 0.815, 0.797, 0.876]

# #### boundary dice score
# eval_scores = [0.394, 0.432, 0.429, 0.501]

# #### instance msa
# eval_scores = [0.098, 0.212, 0.259, 0.198]


# fig, ax = plt.subplots()

# bar_width = 0.35
# bar_positions = np.arange(len(all_models))

# # Plotting the bars
# bars = ax.bar(bar_positions, eval_scores, bar_width)

# # Adding y-values on the bars
# for bar, y_value in zip(bars, eval_scores):
#     ax.text(bar.get_x() + bar.get_width() / 2, y_value, f'{y_value:.3f}', ha='center', va='bottom')

# # Adding labels and title
# ax.set_xticks(bar_positions)
# ax.set_xticklabels(all_models)
# ax.set_xlabel('Models')
# ax.set_ylabel('Dice Score') #Mean Segmentaion Accuracy /
# ax.set_title('Foreground Segmentation Evaluation for MCF7 Cell Type')

# # Show the plot
# plt.show()
# plt.savefig('/home/nimmahen/code/plots/last_plots/MCF7 foreground MSA 4 models.png')




# #### MCF7 semantic

# foreground_scores = [0.788, 0.815, 0.797, 0.876]
# boundary_scores = [0.394, 0.432, 0.429, 0.501]

# bar_width = 0.35
# bar_positions = np.arange(len(all_models))

# fig, ax = plt.subplots()

# # Plotting foreground scores
# bars1 = ax.bar(bar_positions - bar_width/2, foreground_scores, bar_width, label='Foreground')

# # Plotting boundary scores
# bars2 = ax.bar(bar_positions + bar_width/2, boundary_scores, bar_width, label='Boundary')

# # Adding y-values on the bars for foreground scores
# for bar, y_value in zip(bars1, foreground_scores):
#     ax.text(bar.get_x() + bar.get_width() / 2, y_value, f'{y_value:.3f}', ha='center', va='bottom')

# # Adding y-values on the bars for boundary scores
# for bar, y_value in zip(bars2, boundary_scores):
#     ax.text(bar.get_x() + bar.get_width() / 2, y_value, f'{y_value:.3f}', ha='center', va='bottom')

# # Adding labels and title
# ax.set_xticks(bar_positions)
# ax.set_xticklabels(all_models)
# ax.set_xlabel('Models')
# ax.set_ylabel('Dice Score')
# ax.set_title('Semantic Segmentation Evaluation for MCF7 Cell Type')
# ax.legend()
# # Adding custom legend
# ax.legend(loc='lower right')

# # Show the plot
# plt.show()
# plt.savefig('/home/nimmahen/code/plots/last_plots/MCF7_foreground_boundary_MSA_4_models.png')




# ########## Cremi 

# # #### boundary dice score
# #eval_scores = [0.706, 0.719, 0.728, 0.471]

# #### instance msa
# eval_scores = [0.381, 0.388, 0.388, 0.343]


# fig, ax = plt.subplots()

# bar_width = 0.35
# bar_positions = np.arange(len(all_models))

# # Plotting the bars
# bars = ax.bar(bar_positions, eval_scores, bar_width)

# # Adding y-values on the bars
# for bar, y_value in zip(bars, eval_scores):
#     ax.text(bar.get_x() + bar.get_width() / 2, y_value, f'{y_value:.3f}', ha='center', va='bottom')

# # Adding labels and title
# ax.set_xticks(bar_positions)
# ax.set_xticklabels(all_models)
# ax.set_xlabel('Models')
# ax.set_ylabel('Mean Segmentaion Accuracy') #Mean Segmentaion Accuracy /Dice Score
# ax.set_title('Instance Segmentation Evaluation for Cremi Dataset')

# # Show the plot
# plt.show()
# plt.savefig('/home/nimmahen/code/plots/last_plots/Cremi all average instance MSA 4 models.png')






# ### cremi 30 and all

# ####boundary
# # _30_scores = [0.610, 0.618, 0.616, 0.471]
# # _all_scores = [0.706, 0.719, 0.728, 0.471]

# ####instance
# _30_scores = [0.359, 0.380, 0.383, 0.350]
# _all_scores = [0.381, 0.388, 0.388, 0.343]


# bar_width = 0.35
# bar_positions = np.arange(len(all_models))

# fig, ax = plt.subplots()

# # Plotting foreground scores
# bars1 = ax.bar(bar_positions - bar_width/2, _30_scores, bar_width, label='30 Samples')

# # Plotting boundary scores
# bars2 = ax.bar(bar_positions + bar_width/2, _all_scores, bar_width, label='All Samples')

# # Adding y-values on the bars for foreground scores
# for bar, y_value in zip(bars1, _30_scores):
#     ax.text(bar.get_x() + bar.get_width() / 2, y_value, f'{y_value:.3f}', ha='center', va='bottom')

# # Adding y-values on the bars for boundary scores
# for bar, y_value in zip(bars2, _all_scores):
#     ax.text(bar.get_x() + bar.get_width() / 2, y_value, f'{y_value:.3f}', ha='center', va='bottom')

# # Adding labels and title
# ax.set_xticks(bar_positions)
# ax.set_xticklabels(all_models)
# ax.set_xlabel('Models')
# ax.set_ylabel('Mean Segmentaion Accuracy')
# ax.set_title('Instance Segmentation Evaluation for Cremi Dataset')
# ax.legend()
# # Adding custom legend
# ax.legend(loc='lower right')

# # Show the plot
# plt.show()
# plt.savefig('/home/nimmahen/code/plots/last_plots/Cremi_fraction_MSA_4_models.png')
