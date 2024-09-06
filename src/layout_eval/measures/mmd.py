import numpy as np
from tqdm import tqdm


def compute_xy(data1, data2, measure_function, disable_parallel=False):
    n1, n2 = len(data1), len(data2)
    ii, jj = np.meshgrid(range(n1), range(n2))
    ii, jj = ii.flatten(), jj.flatten()
    if disable_parallel:
        xy = 0
        for i, j in tqdm(zip(ii, jj)):
            xy += measure_function(data1[i], data2[j])
    else:
        from joblib import Parallel, delayed

        xy = Parallel(n_jobs=-1)(
            delayed(measure_function)(data1[i], data2[j]) for i, j in tqdm(zip(ii, jj))
        )
        xy = np.sum(xy)
    return xy


def compute_xx(data, measure_function, disable_parallel=False):
    n = len(data)
    ii, jj = np.meshgrid(range(n), range(n))
    ii, jj = ii.flatten(), jj.flatten()
    if disable_parallel:
        xx = 0
        for i, j in tqdm(zip(ii, jj)):
            if (
                i < j
            ):  # only compute upper triangle as measure function is symmetric and diagonal is 0
                xx += measure_function(data[i], data[j])
        xx *= 2
    else:
        from joblib import Parallel, delayed

        xx = Parallel(n_jobs=-1)(
            delayed(measure_function)(data[i], data[j])
            for i, j in tqdm(zip(ii, jj))
            if i < j
        )
        xx = np.sum(xx) * 2
    return xx


def convert_emd_to_affinity(emd, sigma=1.0):
    return np.exp(-emd / sigma)


def estimate_mmd(xx, yy, xy):
    # only use upper triangle (suppose it is symmetric)
    xx = np.mean(xx[np.triu_indices(len(xx))])
    yy = np.mean(yy[np.triu_indices(len(yy))])
    xy = np.mean(xy)
    return xx + yy - 2 * xy
