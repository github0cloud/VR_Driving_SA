import matplotlib.pyplot as plt
import numpy as np

groups = ['Females', 'Males']
values = [5, 9]
colors = ['black', 'white']
ages = np.arange(21, 26)
mean_age = 21.3
std_dev = 1.07
plt.figure()
wedges, texts, autotexts = plt.pie(values, labels=groups, colors=colors, startangle=90, autopct='%1.1f%%', wedgeprops=dict(edgecolor='gray'))
for i, autotext in enumerate(autotexts):
    if colors[i] == 'black':
        autotext.set_color('white')
    else:
        autotext.set_color('black')
plt.title('Group Composition by Gender')
plt.show()
plt.figure()
ages_data = np.random.normal(mean_age, std_dev, 100)  
plt.boxplot(ages_data, patch_artist=True,
            boxprops=dict(facecolor='black', color='white'),
            medianprops=dict(color='white'))
plt.ylabel('Age Distribution')
plt.title('Age Distribution Boxplot')
plt.show()