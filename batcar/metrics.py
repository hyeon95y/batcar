def str_to_agg_func(agg_name):
    import numpy as np

    if agg_name == 'mean':
        return np.mean
    elif agg_name == 'std':
        return np.std
    else:
        raise ValueError('agg_name: {agg_name} is not supported.')