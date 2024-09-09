import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import statsmodels.api as sm
from sub_functions import corla_cofi
from sub_functions import outlier_detection
from scipy.spatial.distance import cdist
import exceptiongroup

def show(x_list, Y_list, score_index):
    corla_cofi.print_CC_pv(x_list, Y_list)
    plt.scatter(x_list, Y_list, color='#AF5DA8', label='Data Points')
    x = np.array(x_list)
    Y = np.array(Y_list)
    model_orign = sm.OLS(Y, x).fit()

    draw_x = np.linspace(0, 1, 10)
    predictions = model_orign.predict(draw_x)
    plt.plot(draw_x, predictions, color='#6EB138', linewidth=2, label='Linear Regression')
   
    if score_index != 4:
        plt.xlabel("L{} score".format(score_index))
    else:
        plt.xlabel("Overall score")
    plt.ylabel("Reciprocal of average acceleration")

    r_squared_val = model_orign.rsquared
    f_statistic_val = model_orign.fvalue
    p_value_val = model_orign.f_pvalue
    r_squared = "R-squared: {:.3f}".format(r_squared_val)
    f_statistic = "F-statistic: {:.3f}".format(f_statistic_val)
    p_value = "p (F-statistic): {:.2e}".format(p_value_val)
    stats_text = "\n".join([r_squared, f_statistic, p_value])
    plt.text(0.02, 0.98, stats_text, horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes, fontsize=10, color='black')
    plt.show()
    print("Regression Function: y = {}x".format(model_orign.params[0]))

