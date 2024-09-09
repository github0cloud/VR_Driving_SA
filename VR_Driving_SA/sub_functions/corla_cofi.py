from scipy.stats import spearmanr, kendalltau, pearsonr
import tomllib
import exceptiongroup

def print_CC_pv(x,y):
    print(spearmanr(x,y))
    print(kendalltau(x,y))
    print("pearsonrResult(correlation={:.12f}, pvalue={:.12f})".format(pearsonr(x,y)[0], pearsonr(x,y)[1]))
