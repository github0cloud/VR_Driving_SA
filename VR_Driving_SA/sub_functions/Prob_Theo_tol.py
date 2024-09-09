import pandas as pd
import tomllib
import exceptiongroup
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

def Prob_densi_rto(data, drop_record = []): 
	data = np.array(data)
	# Calculate the joint probability density
	keep_indices = np.array([i for i in range(len(data)-1) if i not in drop_record])
	pre_data = data[:-1][keep_indices]
	now_data = data[1:][keep_indices]
	joint_kde = stats.gaussian_kde([pre_data, now_data])
	x_kde = stats.gaussian_kde(pre_data)
	y_kde = stats.gaussian_kde(now_data)
	p_pre = x_kde(pre_data)
	p_now = y_kde(now_data)
	p_joint = joint_kde(np.c_[pre_data, now_data].T)
	# Prob_densi_rto = np.sum(p_joint * np.log2(p_joint/(p_pre)))
	# Prob_densi_rto = np.sum(p_joint * np.log2(p_joint/(p_now)))
	Prob_densi_rto = np.sum(p_joint * np.log2(p_joint/(p_pre*p_now)))
	# Prob_densi_rto = np.average(p_joint * np.log2(p_joint/(p_pre*p_now)))
	# Prob_densi_rto = np.sum(np.log2(p_joint/(p_pre*p_now)))
	return Prob_densi_rto