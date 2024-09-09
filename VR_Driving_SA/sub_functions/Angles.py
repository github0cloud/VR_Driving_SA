import numpy as np

def measure_rotation(angle1, angle2):
    diff = (angle2 - angle1) % 360
    if diff > 180:
        diff = 360 - diff
    return abs(diff)*np.sign(angle2 - angle1)
