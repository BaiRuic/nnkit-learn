import random
import math

seed = 0
def set_seed(val):
    global seed
    seed = val



def random_array(*shape, distribution='uniform', params=None, dtype='float'):
    random.seed(seed)
    if params is None:
        params = []
    dist_func = getattr(random, distribution)
    if dtype not in ['int', 'float']:
        raise ValueError('Unknown data type: ' + dtype)
    if len(shape) == 1:
        if dtype == 'int':
            return [int(dist_func(*params)) for _ in range(shape[0])]
        else:
            return [float(dist_func(*params)) for _ in range(shape[0])]
    else:
        return [random_array(*shape[1:], distribution=distribution, params=params, dtype=dtype) for _ in range(shape[0])]

"""
def random_array(*shape, distribution='uniform', params=None):
    if params is None:
        params = []
    if distribution == 'uniform':
        if len(shape) == 1:
            a, b = params
            return [random.uniform(a, b) for _ in range(shape[0])]
        else:
            return [random_array(*shape[1:], distribution=distribution, params=params) for _ in range(shape[0])]
    elif distribution == 'normal':
        if len(shape) == 1:
            mu, sigma = params
            return [random.normalvariate(mu, sigma) for _ in range(shape[0])]
        else:
            return [random_array(*shape[1:], distribution=distribution, params=params) for _ in range(shape[0])]
    elif distribution == 'exponential':
        if len(shape) == 1:
            lambd = params[0]
            return [random.expovariate(lambd) for _ in range(shape[0])]
        else:
            return [random_array(*shape[1:], distribution=distribution, params=params) for _ in range(shape[0])]
    elif distribution == 'lognormal':
        if len(shape) == 1:
            mu, sigma = params
            return [math.exp(random.normalvariate(mu, sigma)) for _ in range(shape[0])]
        else:
            return [random_array(*shape[1:], distribution=distribution, params=params) for _ in range(shape[0])]
    else:
        raise ValueError('Unknown distribution: ' + distribution)
"""