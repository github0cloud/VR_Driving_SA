import numpy as np


# MinMaxScaler for score, range from 1 to 10
def minMax_Scaler(input_arr):
    output_arr = []
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)
    for x in input_arr:
        output_arr.append(((float(x - min_val) / (max_val - min_val)) * 9) + 1)
    return output_arr