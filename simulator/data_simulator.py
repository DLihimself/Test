import copy
from scipy.stats import beta
# 光纤类型
FIBER_TYPE = ['SMF', 'MMF', 'PMF', 'DSF', 'AGF']

# 编码类型
ENCODING_TYPE = ['DM', 'EM', 'CM', 'DSSS']

# 特征列表
FULL_FEATURE_COL = [ 
    'signalStrength', 
    'distance', 
    'fiberType', 
    'frequency', 
    'temperature', 
    'humidity', 
    'encoding', 
    'wavelength', 
    'createDtm']

TEST_FEATURE_COL = [ 
    'signalStrength', 
    'distance', 
    'fiberType', 
    'frequency', 
    'temperature', 
    'humidity', 
    'encoding', 
    'wavelength'] # 测试数据的特征，createDtm不需要

LOWACC_FEATURE_COL = [ 
    'signalStrength', 
    'distance', 
    'fiberType', 
    'frequency', 
    'temperature', 
    'humidity', 
    'encoding', 
    'wavelength']

FEATURE_COL = [ 
    'signalStrength', 
    'distance', 
    'frequency', 
    'temperature', 
    'humidity'] # 用于训练的特征

# 特征值的取值范围
FEATURE_RANGE = {
    'signalStrength': [0, 1], 
    'distance': [10, 120], 
    'frequency': [10, 800], 
    'temperature': [-40, 85], 
    'humidity': [0, 100],
}


NORM = 'normal'
ABNR = 'abnormal'

FEATURE_DIST = {
    'signalStrength': {
        NORM: [8, 1],
        ABNR: [1, 8]},
    'distance': {
        NORM: [1, 8],
        ABNR: [8, 1]},
    'frequency': {
        NORM: [1, 8],
        ABNR: [8, 1]},
    'temperature': {
        NORM: [10,  10],
        ABNR: [0.4, 0.4]},
    'humidity': {
        NORM: [1, 8],
        ABNR: [8, 1]},
}


def normalized(dt, loc, scale):
    return (dt - loc) / scale

def denormalized(dt, loc, scale):
    return dt * scale + loc

def get_normalized_data(data):
    data_scaled = copy.deepcopy(data)
    for fname in FEATURE_COL:
        r_low, r_up = FEATURE_RANGE[fname]
        loc = r_low
        scale = r_up - r_low
        data_scaled[fname] = normalized(data[fname], loc, scale)

    return data_scaled

def get_denormalized_data(data):
    data_inversed = copy.deepcopy(data)
    for fname in FEATURE_COL:
        r_low, r_up = FEATURE_RANGE[fname]
        loc = r_low
        scale = r_up - r_low
        data_inversed[fname] = denormalized(data[fname], loc, scale)

    return data_inversed

def check_full_data(data):
    is_data_valid = True
    invalid_features = []
    for fname in FULL_FEATURE_COL:
        fvalue = data[fname]
        if fname in FEATURE_COL:
            r_low, r_up = FEATURE_RANGE[fname]
            if (fvalue.item() < r_low) or (fvalue.item() > r_up):
                invalid_features.append(fname)

        if fname == 'fiberType':
            if fvalue.item() not in FIBER_TYPE:
                invalid_features.append(fname)

        if fname == 'encoding':
            if fvalue.item() not in ENCODING_TYPE:
                invalid_features.append(fname)

    if len(invalid_features) > 0:
        is_data_valid = False

    return is_data_valid, invalid_features


def simulate_feature_value(fname, n_samples, flabel=NORM):
    a, b = FEATURE_DIST[fname][flabel]
    r_low, r_up = FEATURE_RANGE[fname]
    f_loc = r_low
    f_scale = r_up - r_low
    sm_fdist = beta(a, b, loc=f_loc, scale=f_scale)
    sm_fsamples = sm_fdist.rvs(size=n_samples)

    return sm_fdist, sm_fsamples


