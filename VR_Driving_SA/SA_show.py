import pandas as pd
import numpy as np
import tomllib
import matplotlib.pyplot as plt
import csv
import exceptiongroup
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import statsmodels.api as sm
from sub_functions import corla_cofi
from scipy.spatial.distance import cdist

csv_path = 'result\\result.csv'
avg_MSaclrt_list = []
L1_detctScor_list = []
L2_cmprehScor_list = []
L3_prdctScor_list = []
overallScore_list = []

with open(csv_path, 'r', newline='') as csvfile:
    csvreader = csv.reader(csvfile)
    headers = next(csvreader) 
    for row in csvreader:
        avg_MSaclrt_list.append(float(row[0]))
        L1_detctScor_list.append(float(row[1]))
        L2_cmprehScor_list.append(float(row[2]))
        L3_prdctScor_list.append(float(row[3]))
        overallScore_list.append(float(row[4]))

driv_profom = avg_MSaclrt_list
L1_model = LinearRegression()
L1_model.fit(np.array(L1_detctScor_list).reshape(-1, 1), np.array(driv_profom))
slope = L1_model.coef_[0]
intercept = L1_model.intercept_
predictions = L1_model.predict(np.array(L1_detctScor_list).reshape(-1, 1))
print("------L1: Detection Score------:")
corla_cofi.print_CC_pv(L1_detctScor_list,driv_profom)
plt.scatter(L1_detctScor_list, driv_profom, color='blue', label='Data Points')
plt.plot(L1_detctScor_list, predictions, color='red', linewidth=2, label='Linear Regression')
plt.title("L1: Detection Score")
plt.xlabel("detection score")
plt.ylabel("average acceleration reciprocal")
plt.legend()
plt.show()
print(f"linear_regression_L1_slope={slope}, intercept={intercept}" )

L2_model = LinearRegression()
L2_model.fit(np.array(L2_cmprehScor_list).reshape(-1, 1), np.array(driv_profom))
slope = L2_model.coef_[0]
intercept = L2_model.intercept_
driv_profomance = avg_MSaclrt_list
predictions = L2_model.predict(np.array(L2_cmprehScor_list).reshape(-1, 1))
print("------L2: Comprehend Score------:")
corla_cofi.print_CC_pv(L2_cmprehScor_list,driv_profom)
plt.scatter(L2_cmprehScor_list, driv_profom, color='blue', label='Data Points')
plt.plot(L2_cmprehScor_list, predictions, color='red', linewidth=2, label='Linear Regression')
plt.title("L2: Comprehend Score")
plt.xlabel("comprehend score")
plt.ylabel("average acceleration reciprocal")
plt.legend()
plt.show()
print(f"linear_regression_L2_slope={slope}, intercept={intercept}")

L3_model = LinearRegression()
L3_model.fit(np.array(L3_prdctScor_list).reshape(-1, 1), np.array(driv_profom))
slope = L3_model.coef_[0]
intercept = L3_model.intercept_
predictions = L3_model.predict(np.array(L3_prdctScor_list).reshape(-1, 1))
print("------L3: Predict Score------:")
corla_cofi.print_CC_pv(L3_prdctScor_list,driv_profom)
print(f"linear_regression_L3_slope={slope}, intercept={intercept}")
plt.scatter(L3_prdctScor_list, driv_profom, color='blue', label='Data Points')
plt.plot(L3_prdctScor_list, predictions, color='red', linewidth=2, label='Linear Regression')
plt.title("L3: Predict Score")
plt.xlabel("predict score")
plt.ylabel("average acceleration reciprocal")
plt.legend()
plt.show()

overall_features = np.column_stack((L1_detctScor_list, L2_cmprehScor_list, L3_prdctScor_list))
overall_pca = PCA(n_components=3)
overall_pca.fit(overall_features)
overall_pca.components_ = -overall_pca.components_
PCA_List =  list(overall_pca.transform(overall_features))
overallScore_list = [arr[0] for arr in PCA_List]
overall_model = LinearRegression()
overall_model.fit(np.array(overallScore_list).reshape(-1, 1), np.array(driv_profom))
slope = overall_model.coef_[0]
intercept = overall_model.intercept_
predictions = overall_model.predict(np.array(overallScore_list).reshape(-1, 1))
print("------overall Score------:")
corla_cofi.print_CC_pv(overallScore_list,driv_profom)
plt.scatter(overallScore_list, driv_profom, color='blue', label='Data Points')
plt.plot(overallScore_list, predictions, color='red', linewidth=2, label='Linear Regression')
plt.title("overall Score")
plt.xlabel("overall score")
plt.ylabel("average acceleration reciprocal")
plt.legend()
plt.show()
print(f"linear_regression_overall_slope={slope}, intercept={intercept}")
print("")
print("")

print("-explained variance-:")
print(overall_pca.explained_variance_ratio_)
sizes = overall_pca.explained_variance_ratio_
colors = ['lightcoral', 'lightskyblue', 'lightgreen']
labels = ['1st PC', '2nd PC', '3rd PC']
plt.pie(sizes, colors=colors, labels = labels, autopct='%1.1f%%')
plt.title('Pie Chart')
plt.show()



