import numpy as np


def gaussian_weighter(x,mu,sigma):
    gauss_weighted = x * np.exp(-((x - mu)**2) / (2*sigma**2)) 
    return gauss_weighted

