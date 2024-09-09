import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sub_functions import vec_diff
from sub_functions import corla_cofi
from sub_functions import Prob_Theo_tol
from sub_functions import scalers

avg_MSaclrt_list = []
avg_MSaclReci_list = []
avg_frq= 51.34
objPrcpFreq_list = []
objCount_list = []
wehtDetctScor_list = []
avgDwellTime_list = []
aveGazeRate_list = []
timeDura_list = []
sceNams_list = ['Day.xlsx', 'DuskOff.xlsx', 'DuskOn.xlsx', 'Night.xlsx']
for sceNam in sceNams_list:
    for i in range(1,15):
        file_path = 'data_gaze\\'+str(i) + sceNam
        dataSet = pd.read_excel(file_path) 
        datLen = len(dataSet)
        objPosi_cols = ['SystemTime', 'NowSsceneName', 'Hitname', 'HitPointX', 'HitPointY', 'HitPointZ']
        objPosi_dict = dataSet[objPosi_cols].to_dict('list')
        eyePosi_cols = ['SystemTime', 'NowSsceneName', 'Hitname', 'CameraPosX', 'CameraPosY', 'CameraPosZ']
        eyePosi_dict = dataSet[eyePosi_cols].to_dict('list')
        gazeLoclPosi_cols = ['SystemTime', 'NowSsceneName', 'Hitname', 'VecX', 'VecY', 'VecZ']
        gazeLoclPosi_dict = dataSet[gazeLoclPosi_cols].to_dict('list')
        carRec_cols = ['SystemTime', 'NowSsceneName', 'CarDirX', 'CarDirY', 'CarDirZ', 'speed']
        carRec_dict = dataSet[carRec_cols].to_dict('list')
        timLen = objPosi_dict['SystemTime'][-1] - objPosi_dict['SystemTime'][0]
        objCount = len(set(objPosi_dict['Hitname']))
        objCount_list.append(objCount/datLen)
        MSaclrt_list = [((carRec_dict['speed'][i+1] - carRec_dict['speed'][i])*avg_frq)**2 for i in range(datLen-1)]
        avg_MSaclrt = np.mean(MSaclrt_list)
        timLen = objPosi_dict['SystemTime'][-1] - objPosi_dict['SystemTime'][0]
        objNam_set = set(objPosi_dict['Hitname'])
        avg_MSaclrt_list.append(avg_MSaclrt)
        avg_MSaclReci_list.append(1/avg_MSaclrt)
        gazPosi_dict = gazeLoclPosi_dict
        for i in range(datLen):
            gazPosi_dict['VecX'][i] = gazeLoclPosi_dict['VecX'][i] + eyePosi_dict['CameraPosX'][i]
            gazPosi_dict['VecY'][i] = gazeLoclPosi_dict['VecY'][i] + eyePosi_dict['CameraPosY'][i]
            gazPosi_dict['VecZ'][i] = gazeLoclPosi_dict['VecZ'][i] + eyePosi_dict['CameraPosZ'][i]
        objPrcpCont = 0
        stareLen_list = []
        stareNam_list = []
        stareLen = 0
        stp = 1
        for i in range(1,datLen):
            if objPosi_dict['Hitname'][i] == objPosi_dict['Hitname'][i-1]:
                time_pass = carRec_dict['SystemTime'][i] - carRec_dict['SystemTime'][i-1]
                stareLen+=time_pass
            if objPosi_dict['Hitname'][i] != objPosi_dict['Hitname'][i-1]:
                objPrcpCont+=1
                stareLen_list.append(stareLen)
                stareNam_list.append(objPosi_dict['Hitname'][i])
                stareLen = 0 
        avgDwellTime_list.append(np.average(stareLen_list))
        aveGazeRate_list.append(np.average(objPrcpCont/timLen))
    print(sceNam,"done")

avg_MSaclrt_list = scalers.minMax_Scaler(avg_MSaclrt_list)
avg_MSaclReci_list = scalers.minMax_Scaler(avg_MSaclReci_list)
driv_profom = avg_MSaclReci_list
avgDwellTime_list = scalers.minMax_Scaler(avgDwellTime_list)
aveGazeRate_list = scalers.minMax_Scaler(aveGazeRate_list)
print("------aveGazeRate_list------:")
corla_cofi.print_CC_pv(aveGazeRate_list,driv_profom)
plt.scatter(aveGazeRate_list, driv_profom)
plt.xlabel("Angel probability density correlation")
plt.ylabel("Average acceleration")
plt.show()
print("------avgDwellTime_list------:")
corla_cofi.print_CC_pv(avgDwellTime_list,driv_profom)
plt.scatter(avgDwellTime_list, driv_profom)
plt.xlabel("Length probability density correlation")
plt.ylabel("Average acceleration")
plt.show()
