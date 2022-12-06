import re
import math
import numpy as np
import itertools


def parse_yaml_value(arguments, param, param_type):
    """
    Parse config from yaml
    """
    value_reg = r"[0-9]+\.?[0-9]*"
    unit_reg = r"[a-z][A-Z]*"
    raw = arguments[f"{param}_{param_type}"]

    if type(raw) != str:
        value = raw
        unit = ''
    else:
        value = float(re.findall(value_reg, raw)[0])
        unit = re.findall(unit_reg, raw)[0]

    return value, unit


def get_circuit_params(arguments, param):
    """
    Return Circuit parameters
    """
    start, start_unit = parse_yaml_value(arguments, param, 'start')
    stop, stop_unit = parse_yaml_value(arguments, param, 'stop')
    change, change_unit = parse_yaml_value(arguments, param, 'change')

    assert (start_unit == stop_unit == change_unit), f"not the same for all parts of range: parameter: " \
                                                     f"{param}, start {stop_unit}, stop {stop_unit}, " \
                                                     f"change {change_unit} "
    return start, stop, change, stop_unit


def prepare_data(parameter_list, arguments):
    "convert yaml range to numpy array of strings "
    all_ranges = []
    for param in parameter_list:
        start, stop, change, unit = get_circuit_params(arguments, param)
        param_range = np.arange(0, math.ceil((stop - start) / change) + 1) * change + start
        param_range = list(map(lambda x: str(np.round(x, 3)) + unit, param_range))
        all_ranges.append(list(param_range))

    train_data = np.array(list(itertools.product(*all_ranges)))
    print(f"training data size = {train_data.shape}")
    return train_data