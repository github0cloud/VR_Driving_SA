import csv
import exceptiongroup
import pandas as pd
import numpy as np
import tomllib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sub_functions import vec_diff
from sub_functions import corla_cofi
from sub_functions import Prob_Theo_tol
from sub_functions import scalers
from sub_functions import weighters
from sklearn.linear_model import LinearRegression
import math

avg_MSaclrt_list = []
avg_MSaclReci_list = []
L1_detctScor_list = []
L2_cmprehScor_list = []
agl_densi_cor_list = []
len_densi_cor_list = []
L3_prdctScor_list = []
avg_frq= 51.34
sceNams_list = ['Day.xlsx', 'DuskOff.xlsx', 'DuskOn.xlsx', 'Night.xlsx']

for sceNam in sceNams_list:
    for i in range(1,15):
       #################### data loading ####################
        try:
            file_path = 'data_gaze\\'+str(i) + sceNam
            dataSet = pd.read_excel(file_path)
        except FileNotFoundError:
            print("The Unity project files utilized to construct the VR experimental environment, along with the collected experimental data, are owned by the laboratory and are currently not available. For inquiries regarding access, please contact yjiang485@163.com.")
            print("Please execute the SA_show.py script to view the results of the SA analysis for the VR driving experiments.")
            exit()
       

        datLen = len(dataSet)
        objPosi_cols = ['SystemTime', 'NowSsceneName', 'Hitname', 'HitPointX', 'HitPointY', 'HitPointZ']
        objPosi_dict = dataSet[objPosi_cols].to_dict('list')
        intr = [abs(objPosi_dict['SystemTime'][i+1] - objPosi_dict['SystemTime'][i]) for i in range(datLen-1)]
        avg_intr = np.mean(intr)
        eyePosi_cols = ['SystemTime', 'NowSsceneName', 'Hitname', 'CameraPosX', 'CameraPosY', 'CameraPosZ']
        eyePosi_dict = dataSet[eyePosi_cols].to_dict('list')
        gazeLoclPosi_cols = ['SystemTime', 'NowSsceneName', 'Hitname', 'VecX', 'VecY', 'VecZ']
        gazeLoclPosi_dict = dataSet[gazeLoclPosi_cols].to_dict('list')
        carRec_cols = ['SystemTime', 'NowSsceneName', 'CarDirX', 'CarDirY', 'CarDirZ', 'speed']
        carRec_dict = dataSet[carRec_cols].to_dict('list')
        MSaclrt_list = [((carRec_dict['speed'][i+1] - carRec_dict['speed'][i])*avg_frq)**2 for i in range(datLen-1)]
        avg_MSaclrt = np.mean(MSaclrt_list)
        timLen = objPosi_dict['SystemTime'][-1] - objPosi_dict['SystemTime'][0]
        objNam_set = set(objPosi_dict['Hitname'])
        avg_MSaclrt_list.append(avg_MSaclrt)
        avg_MSaclReci_list.append(1/avg_MSaclrt)
        gazPosi_dict = gazeLoclPosi_dict

        #################### L1 & L2 data preparation ####################
        for i in range(datLen):
            gazPosi_dict['VecX'][i] = gazeLoclPosi_dict['VecX'][i] + eyePosi_dict['CameraPosX'][i]
            gazPosi_dict['VecY'][i] = gazeLoclPosi_dict['VecY'][i] + eyePosi_dict['CameraPosY'][i]
            gazPosi_dict['VecZ'][i] = gazeLoclPosi_dict['VecZ'][i] + eyePosi_dict['CameraPosZ'][i]
        objPrcpCont = 0
        stareLen_list = []
        stareNam_list = []
        stareLen = 1
        stp = 1
        for i in range(1,datLen):
            if objPosi_dict['Hitname'][i] == objPosi_dict['Hitname'][i-1]:
                stareLen+=1
            if objPosi_dict['Hitname'][i] != objPosi_dict['Hitname'][i-1]:
                objPrcpCont+=1
                stareLen_list.append(stareLen)
                stareNam_list.append(objPosi_dict['Hitname'][i])
                stareLen = 1
        #################### L1 detection score calc ####################
        L1_detctScor_list.append((objPrcpCont/datLen)*np.average(stareLen_list))
        weightedObjPrcpCont = 0
        for stareLen in stareLen_list:
            weightedScore =  weighters.gaussian_weighter(stareLen,np.average(stareLen_list),np.std(stareLen_list))*stareLen
            weightedObjPrcpCont +=  weightedScore
        #################### L2 comprehension score calc ####################
        L2_cmprehScor_list.append(weightedObjPrcpCont/datLen)


        #################### L3 data preparation ####################
        aglDif_list = []
        lenDif_list = []
        drop_record = []
        drop_index = 0
        stp = 1
        last_obj_name = gazPosi_dict['Hitname'][0]
        for i in range(stp,datLen-stp):
            if (gazPosi_dict['Hitname'][i-stp] == gazPosi_dict['Hitname'][i]) and (gazPosi_dict['Hitname'][i] == gazPosi_dict['Hitname'][i+stp]):                
                las_vec = np.array([gazPosi_dict['VecX'][i]-gazPosi_dict['VecX'][i-stp],
                                    gazPosi_dict['VecY'][i]-gazPosi_dict['VecY'][i-stp],
                                    gazPosi_dict['VecZ'][i]-gazPosi_dict['VecZ'][i-stp]]) 
                nex_vec = np.array([gazPosi_dict['VecX'][i+stp]-gazPosi_dict['VecX'][i],
                                    gazPosi_dict['VecY'][i+stp]-gazPosi_dict['VecY'][i],
                                    gazPosi_dict['VecZ'][i+stp]-gazPosi_dict['VecZ'][i]]) 
                gaze_vec = np.array([gazeLoclPosi_dict['VecX'][i], gazeLoclPosi_dict['VecY'][i], gazeLoclPosi_dict['VecZ'][i]])               
                aglDif, lenDif = vec_diff.agl_len(las_vec, nex_vec, gaze_vec)
                aglDif_list.append(aglDif)
                lenDif_list.append(-np.sign(lenDif)*np.log10(abs(lenDif))) 
                if last_obj_name != gazPosi_dict['Hitname'][i-stp] or gazPosi_dict['Hitname'][i-stp] =="far":
                    drop_record.append(drop_index)
                    last_obj_name = gazPosi_dict['Hitname'][i-stp]
                drop_index = drop_index+1
        agl_densi_cor_list.append(Prob_Theo_tol.Prob_densi_rto(aglDif_list, drop_record)) 
        len_densi_cor_list.append(Prob_Theo_tol.Prob_densi_rto(lenDif_list, drop_record)) 
    print(sceNam,"done")

avg_MSaclrt_list = scalers.minMax_Scaler(avg_MSaclrt_list)
avg_MSaclReci_list = [1/item for item in avg_MSaclrt_list]
avg_MSaclReci_list = scalers.minMax_Scaler(avg_MSaclReci_list)
driv_profom = avg_MSaclReci_list
L1_detctScor_list = scalers.minMax_Scaler(L1_detctScor_list)
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
L2_cmprehScor_list = scalers.minMax_Scaler(L2_cmprehScor_list)
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
#################### L3 prediction score calc ####################
agl_densi_cor_list = scalers.minMax_Scaler(agl_densi_cor_list)
len_densi_cor_list = scalers.minMax_Scaler(len_densi_cor_list)
L3_features = np.column_stack((agl_densi_cor_list, len_densi_cor_list))
L3_pca = PCA(n_components=1)
L3_pca.fit(L3_features)
L3_prdctScor_list = list(L3_pca.transform(L3_features).flatten())
L3_prdctScor_list = scalers.minMax_Scaler(L3_prdctScor_list)
L3_model = LinearRegression()
L3_model.fit(np.array(L3_prdctScor_list).reshape(-1, 1), np.array(driv_profom))
slope = L3_model.coef_[0]
intercept = L3_model.intercept_
predictions = L3_model.predict(np.array(L3_prdctScor_list).reshape(-1, 1))
print("------L3: Predict Score------:")
corla_cofi.print_CC_pv(L3_prdctScor_list,driv_profom)
print("------L3: Dir Predict Score------:")
corla_cofi.print_CC_pv(agl_densi_cor_list,driv_profom)
print("------L3: Spd Predict Score------:")
corla_cofi.print_CC_pv(len_densi_cor_list,driv_profom)
print(f"linear_regression_L3_slope={slope}, intercept={intercept}")
plt.scatter(L3_prdctScor_list, driv_profom, color='blue', label='Data Points')
plt.plot(L3_prdctScor_list, predictions, color='red', linewidth=2, label='Linear Regression')
plt.title("L3: Predict Score")
plt.xlabel("predict score")
plt.ylabel("average acceleration reciprocal")
plt.legend()
plt.show()
#################### overall score calc ####################
overall_features = np.column_stack((L1_detctScor_list, L2_cmprehScor_list, L3_prdctScor_list))
overall_pca = PCA(n_components=3)
overall_pca.fit(overall_features)
overall_pca.components_ = -overall_pca.components_
PCA_List =  list(overall_pca.transform(overall_features))
overallScore_list = [arr[0] for arr in PCA_List]
overallScore_list = scalers.minMax_Scaler(overallScore_list)
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
data_store = list(zip(driv_profom, L1_detctScor_list, L2_cmprehScor_list, L3_prdctScor_list, overallScore_list))
csv_path = 'result\\result.csv'
with open(csv_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerow(['driv_profom', 'L1_detctScor_list','L2_cmprehScor_list','L3_prdctScor_list', 'overallScore_list'])
    csvwriter.writerows(data_store)


