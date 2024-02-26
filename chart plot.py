import matplotlib.pyplot as plt
import numpy as np

# Your data
three_models = ["UNETR_sc", "UNETR_sam_vit_b", "UNETR_sam_vit_l" "UNET"]

#### foreground dice scores
eval_scores = [(0.776, 0.774), (0.826, 0.845), (0.801, 0.815), (0.882, 0.898)]

#### boundaries dice scores
eval_scores = [(0.382, 0.394), (0.422, 0.453), (0.411, 0.404), (0.498, 0.522)]

#### instance msa
eval_scores = [(0.179, 0.198), (0.238, 0.284), (0.330, 0.332), (0.240, 0.289)]

# Extracting the data for plotting
data = np.array(eval_scores)
bar_width = 0.35
bar_positions = np.arange(len(three_models))

# Plotting
fig, ax = plt.subplots()

legend_labels = ["60samples", "all"]  # Custom labels for the legend

for i in range(len(data[0])):
    bars = ax.bar(bar_positions - bar_width / 2 + i * bar_width, data[:, i], bar_width, label=legend_labels[i])

    # Adding y-values on the bars
    for bar, y_value in zip(bars, data[:, i]):
        ax.text(bar.get_x() + bar.get_width() / 2, y_value, f'{y_value:.3f}', ha='center', va='bottom')

# Adding labels and title
ax.set_xticks(bar_positions)
ax.set_xticklabels(three_models)
ax.set_xlabel('Models')
ax.set_ylabel('Mean Segmentaion Accuracy') ### Dice Score
ax.set_title('LiveCell all average instance MSA')

# Adding custom legend
ax.legend(loc='lower right')

# Show the plot
plt.show()


plt.savefig('/home/nimmahen/code/plots/LiveCell all average instance MSA.png')
