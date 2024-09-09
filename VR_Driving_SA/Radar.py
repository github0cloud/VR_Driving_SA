import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

subj1_max_score = [81, 54, 62, 70, 72, 85, 54, 17, 22, 57, 60]
subj2_min_score = [23, 20, 28, 6, 8, 12, 35, 37, 36, 56, 25]

labels = np.array(['L1', 'L2', 'L3_dir', 'L3_spd', 'L3', 'Overall', 'GTE', 'SGE', 'Gaze rate', 'Dwell time', 'VSE' ])
num_vars = len(labels)

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), subplot_kw=dict(polar=True))

ax1.grid(True, linestyle='--', linewidth=0.5)
ax2.grid(True, linestyle='--', linewidth=0.5)
ax1.spines['polar'].set_visible(False)
ax2.spines['polar'].set_visible(False)

ax1.fill(angles, subj1_max_score, '#AF5DA8', alpha=0.25) 
ax1.plot(np.concatenate((angles, [angles[0]])), np.concatenate((subj1_max_score, [subj1_max_score[0]])), '#AF5DA8', linewidth=2, label='subj1_score') 
ax1.set_yticklabels([])
ax1.set_xticks(angles)
ax1.set_xticklabels(labels, fontsize=12)

ax2.fill(angles, subj2_min_score, '#6EB138', alpha=0.25)
ax2.plot(np.concatenate((angles, [angles[0]])), np.concatenate((subj2_min_score, [subj2_min_score[0]])), '#6EB138', linewidth=2, label='subj2_score')
ax2.set_yticklabels([])
ax2.set_xticks(angles)
ax2.set_xticklabels(labels, fontsize=12)

max_score = max(max(subj1_max_score), max(subj2_min_score))
ax1.set_ylim(0, max_score)
ax2.set_ylim(0, max_score)

plt.show()
