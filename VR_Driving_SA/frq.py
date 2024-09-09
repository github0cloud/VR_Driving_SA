import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sub_functions import Angles
from sub_functions import tim_relat
import os
import csv
from datetime import datetime
        
sceNams_list = ['Day.xlsx', 'DuskOff.xlsx', 'DuskOn.xlsx', 'Night.xlsx']
avg_aclrt_list = []
avg_aclrtReci_list = []
time_interval_list = []

experi_id = 0
for l,sceNam in enumerate(sceNams_list):
    for i in range(1,15):
        file_path = 'data_gaze\\'+str(i) + sceNam
        dataSet = pd.read_excel(file_path) 
        datLen = len(dataSet)
        eye_dir_cols = ['SystemTime']
        time_list = dataSet['SystemTime'].tolist()
        time_interval = [tim_relat.tim_dif(time_list[i], time_list[i+1]) for i in range(datLen-1)]
        print("file_path=",file_path)
        mean = np.mean(time_interval)
        print("mean",mean)
        std = np.std(time_interval)
        print("std",std)
        outlier_index_list = [i for i in range(datLen-1) if tim_relat.tim_dif(time_list[i], time_list[i+1]) > mean + 3*std or tim_relat.tim_dif(time_list[i], time_list[i+1]) < mean - 3*std]
        outlier_value_list = [tim_relat.tim_dif(time_list[i], time_list[i+1]) for i in range(datLen-1) if tim_relat.tim_dif(time_list[i], time_list[i+1]) > mean + 3*std or tim_relat.tim_dif(time_list[i], time_list[i+1]) < mean - 3*std]
        norm_list = [tim_relat.tim_dif(time_list[i], time_list[i+1]) for i in range(datLen-1) if tim_relat.tim_dif(time_list[i], time_list[i+1]) < mean + 3*std and tim_relat.tim_dif(time_list[i], time_list[i+1]) > mean - 3*std]
        print("outlier_index_list=",outlier_index_list)
        print("outlier_value_list=",outlier_value_list)
        time_interval_list.append(np.mean(norm_list))
    print(sceNam," done")
mean = np.mean(time_interval_list)
print("mean",mean)
std = np.std(time_interval_list)
print("std",std)
print("avg_frq=",1000/mean)
# avg_frq= 51.34