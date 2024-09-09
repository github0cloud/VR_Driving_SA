import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable

indicators = ['L1 score', 'L2 score', 'L3_dir', 'L3_spd', 'L3 score', 'Overall score', 'GTE', 'SGE', 'Gaze rate', 'Dwell time', 'VSE']
cc_spearman = [0.31, 0.55, 0.42, 0.67, 0.64, 0.66, 0.17, 0.09, -0.06, 0.09, 0.40]
p_spearman = [0.02, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.05, 0.05, 0.05, 0.05, 0.00001]
cc_kendall = [0.22, 0.39, 0.30, 0.50, 0.46, 0.48, 0.12, 0.06, 0.03, 0.06, 0.29]
p_kendall = [0.02, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.05, 0.05, 0.05, 0.05, 0.0001]
cc_pearson = [0.28, 0.40, 0.38, 0.52, 0.63, 0.60, 0.20, 0.11, 0.18, -0.24, 0.46]
p_pearson = [0.02, 0.0017, 0.0001, 0.0001, 0.0001, 0.0001, 0.05, 0.05, 0.05, 0.05, 0.0001]
colors = {
    'Pearson': ['#216fab', '#508dbe', '#80acd1', '#afcae4'], # b
    'Kendall': ['#52ce1c', '#79de50', '#a0ef84', '#c7ffb8'], # g drk - lgt
    'Spearman': ['#e13737', '#eb7070', '#f4a8a8', '#fee1e1'] # r
}

indicators.reverse()
cc_spearman.reverse()
p_spearman.reverse()
cc_kendall.reverse()
p_kendall.reverse()
cc_pearson.reverse()
p_pearson.reverse()
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.25
bar_positions1 = np.arange(len(indicators))
bar_positions2 = bar_positions1 + bar_width
bar_positions3 = bar_positions2 + bar_width

for i, (cc_values, p_values, method) in enumerate(zip([cc_pearson, cc_kendall, cc_spearman],
                                                      [p_pearson, p_kendall,p_spearman ],
                                                      ['Pearson', 'Kendall', 'Spearman'])):
    color_values = [colors[method][np.digitize(p, [0.001, 0.01, 0.05, 1.0])] for p in p_values]
    if i == 0:
        ax.barh(bar_positions1, cc_values, height=bar_width, color=color_values, label=method)
    elif i == 1:
        ax.barh(bar_positions2, cc_values, height=bar_width, color=color_values, label=method)
    elif i == 2:
        ax.barh(bar_positions3, cc_values, height=bar_width, color=color_values, label=method)

ax.set_yticks(bar_positions2)
ax.set_yticklabels(indicators)
ax.set_xlabel('Correlation Coefficient')
ax.set_title('CC between SA easures and Driving Performance')
sm_spearman = ScalarMappable(cmap=ListedColormap(colors['Spearman']), norm=Normalize(vmin=0, vmax=1))
sm_spearman.set_array([])
sm_kendall = ScalarMappable(cmap=ListedColormap(colors['Kendall']), norm=Normalize(vmin=0, vmax=1))
sm_kendall.set_array([])
sm_pearson = ScalarMappable(cmap=ListedColormap(colors['Pearson']), norm=Normalize(vmin=0, vmax=1))
sm_pearson.set_array([])
cbar1 = plt.colorbar(sm_spearman, ticks=[], orientation='vertical', pad=0)
cbar2 = plt.colorbar(sm_kendall, ticks=[], orientation='vertical', pad=0.02)
cbar3 = plt.colorbar(sm_pearson, ticks=[], orientation='vertical', pad=0.04)
plt.show()

