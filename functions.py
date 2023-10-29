from typing import Union
import numpy as np
from collections import namedtuple

Domain = namedtuple('Domain', ['x_lower_bound', 'x_upper_bound', 'y_lower_bound', 'y_upper_bound'])


DOMAIN = Domain(-5, 5, -5, 5)


def get_f1(x1: Union[float, np.ndarray], y1: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	"""
	Get the value of function f1 in point x1, y1
	"""
	return (x1**2 + y1 - 11)**2 + (x1 + y1**2 - 7)**2


def get_f2(x2: Union[float, np.ndarray], y2: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	"""
	Get the value of function f1 in point x1, y1
	"""
	return 2*(x2**2) + 1.05*(x2**4) + (x2**6)/6 + x2*y2 + y2**2
