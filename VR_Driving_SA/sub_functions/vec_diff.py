import numpy as np
import tomllib
import matplotlib.pyplot as plt
import exceptiongroup

def agl_len(v1, v2, gaze_vec=[0, 0, 1]):
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    cros = np.cross(v1, v2)
    if np.dot(cros, gaze_vec) > 0:
        cros_signed = np.linalg.norm(cros)
    else:
        cros_signed = -np.linalg.norm(cros)
    sin_val = cros_signed / (v1_norm * v2_norm) 
    agl_diff = sin_val
    len_diff = v2_norm - v1_norm
    return agl_diff, len_diff