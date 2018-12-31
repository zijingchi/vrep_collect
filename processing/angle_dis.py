import numpy as np

metrics = [0.5, 2, 1.5, 0.1, 0.1]


def config_dis(config1, config2):
    y = np.multiply(metrics, config2 - config1)
    return np.linalg.norm(y)

