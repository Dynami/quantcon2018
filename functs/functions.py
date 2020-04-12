import numpy as np
import math
import pandas as pd

def cycle(a, len=7):
    return np.cos(2 * np.pi * a / len)


def scale(x: np.array, min=None, max=None):
    _min = min if min is not None else np.min(x)
    _max = max if max is not None else np.max(x)
    # print(_min, _max)
    return 2 * (x - _min) / (_max - _min) - 1


def smoothF(period):
    return 2. / (period + 1)


def highPassFilter(data, period=48):
    a = (0.707 * 2 * math.pi) / period
    alpha1 = 1. + (math.sin(a) - 1.) / math.cos(a)
    b = 1. - alpha1 / 2.
    c = 1. - alpha1

    hp = np.zeros((data.shape[0],));
    for i in range(3, data.shape[0]):
        subset = data[i - 3:i]
        hp[i] = b * b * (subset[-1] - 2 * subset[-2] + subset[-3]) + 2 * c * hp[i - 1] - c * c * hp[i - 2]
    hp = pd.Series(name='hp', index=data.index, data=hp)
    return hp


def lowPassFilter(data, period):
    # lp = np.zeros((data.shape[0],))
    lp = np.copy(data)
    a = smoothF(period)
    a2 = a ** 2
    for i in range(4, data.shape[0]):
        subset = data[i - 4:i]
        lp[i] = (a - 0.25 * a2) * subset[-1] \
                + 0.5 * a2 * subset[-2] \
                - (a - 0.75 * a2) * subset[-3] \
                + 2 * (1 - a) * lp[i - 1] \
                - ((1. - a) ** 2) * lp[i - 2]

    # lp = pd.Series(name='lp', index=data.index, data=lp)
    return lp


def mmi(data):
    '''
    Market Meanness Index
    '''
    data = np.reshape(data, (-1))
    m = np.median(data)
    # print(data[-10:], m)
    length = len(data)
    nh, nl = 0, 0
    for i in range(1, length):
        if (data[i - 1] > m and data[i - 1] > data[i]):
            nl = nl + 1
        elif (data[i - 1] < m and data[i - 1] < data[i]):
            nh = nh + 1

    return 100.0 * (nl + nh) / (length - 1)


def market_meanness_index(data, period):
    mmis = np.ones((period,))
    mmis *= 50
    mmis = mmis.tolist()
    for i in range(period, len(data)):
        subset = data[i - period:i]
        _mmi = mmi(subset)
        mmis.append(_mmi)
    return np.array(mmis)
